"""Backend-neutral Parquet recognition, work planning, and cuDF execution."""

from __future__ import annotations

import hashlib
import json
import numbers
import os
import pickle
import time
import urllib.parse
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Literal, Optional

from ray_data_gpu_fusion._compat import (
    BlockAccessor,
    BlockEntry,
    ParquetDatasource,
    Read,
    RefBundle,
    iterate_with_retry,
    ray,
)
from ray_data_gpu_fusion.runtime import CudfRuntimeContext, FrameStream


READ_PARQUET = "read_parquet"


class ParquetPlanningError(ValueError):
    def __init__(self, reason: str, detail: str):
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}")


@dataclass(frozen=True)
class ScanDescriptor:
    filesystem: Any
    source_kind: Literal["local", "s3"]
    paths: tuple[str, ...]
    listed_file_sizes: tuple[int, ...]
    projection: tuple[str, ...]
    file_schema: Any
    region: Optional[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "paths", tuple(self.paths))
        object.__setattr__(self, "listed_file_sizes", tuple(self.listed_file_sizes))
        object.__setattr__(self, "projection", tuple(self.projection))
        if self.source_kind not in ("local", "s3"):
            raise ValueError("external Parquet source must be local or S3")
        if len(self.paths) != len(self.listed_file_sizes):
            raise ValueError("Parquet paths and listed sizes do not align")
        if any(size < 0 for size in self.listed_file_sizes):
            raise ValueError("listed Parquet sizes must be nonnegative")
        if not self.projection:
            raise ValueError("direct Parquet reads require a nonempty projection")
        if self.source_kind == "s3" and not self.region:
            raise ValueError("direct S3 reads require a region")


@dataclass(frozen=True)
class ScanRecognition:
    descriptor: Optional[ScanDescriptor] = None
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        if (self.descriptor is None) == (self.reason is None):
            raise ValueError("exactly one of descriptor or reason is required")

    @property
    def accepted(self) -> bool:
        return self.descriptor is not None


def _reject(reason: str) -> ScanRecognition:
    return ScanRecognition(reason=reason)


def recognize_scan(logical_op: Any) -> ScanRecognition:
    """Recognize only the stable external descriptor exposed by patched Ray."""

    if type(logical_op) is not Read:
        return _reject("logical_read")
    datasource = logical_op.datasource
    if (
        type(datasource) is not ParquetDatasource
        or logical_op.datasource_or_legacy_reader is not datasource
    ):
        return _reject("parquet_datasource")
    if logical_op.per_block_limit is not None:
        return _reject("per_block_limit")

    result = datasource.get_external_scan_descriptor()
    descriptor = getattr(result, "descriptor", None)
    if descriptor is None:
        return _reject(f"external_scan:{getattr(result, 'reason', 'declined')}")
    try:
        normalized = ScanDescriptor(
            filesystem=descriptor.filesystem,
            source_kind=descriptor.source_kind,
            paths=descriptor.paths,
            listed_file_sizes=descriptor.listed_file_sizes,
            projection=descriptor.projection,
            file_schema=descriptor.file_schema,
            region=descriptor.region,
        )
    except (AttributeError, TypeError, ValueError) as error:
        return _reject(f"invalid_external_scan:{error}")

    if normalized.source_kind == "s3":
        if os.environ.get("AWS_ENDPOINT_URL") or os.environ.get("AWS_ENDPOINT_URL_S3"):
            return _reject("s3_endpoint_override")
        try:
            import botocore.session

            credentials = botocore.session.get_session().get_credentials()
            frozen = credentials.get_frozen_credentials() if credentials else None
            if frozen is None or not frozen.access_key or not frozen.secret_key:
                return _reject("s3_ambient_credentials")
        except Exception:
            return _reject("s3_ambient_credentials")
    return ScanRecognition(descriptor=normalized)


@dataclass(frozen=True)
class SourceIdentity:
    size: int
    marker: str

    def __post_init__(self) -> None:
        if self.size < 0 or not self.marker:
            raise ValueError("source identity requires a size and stable marker")


@dataclass(frozen=True)
class RowGroup:
    path: str
    identity: SourceIdentity
    row_group: int
    num_rows: int
    compressed_bytes: int
    uncompressed_bytes: int


@dataclass(frozen=True)
class FileWork:
    path: str
    identity: SourceIdentity
    row_groups: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "row_groups", tuple(self.row_groups))
        if not self.path or not self.row_groups:
            raise ValueError("file work requires a path and row groups")
        if self.row_groups != tuple(sorted(set(self.row_groups))):
            raise ValueError("row groups must be sorted and unique")


@dataclass(frozen=True)
class ParquetWork:
    work_id: int
    scheme_id: str
    files: tuple[FileWork, ...]
    num_rows: int
    compressed_bytes: int
    uncompressed_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "files", tuple(self.files))
        if self.work_id < 0 or not self.scheme_id or not self.files:
            raise ValueError("Parquet work requires an ID, scheme, and files")


@dataclass(frozen=True)
class FooterSummary:
    row_groups: tuple[RowGroup, ...]
    footer_bytes: int
    elapsed_s: float


@dataclass(frozen=True)
class ParquetPlan:
    target_tasks: int
    scheme_id: str
    work: tuple[ParquetWork, ...]
    footer_bytes: int
    footer_elapsed_s: float


def _split_s3(path: str) -> tuple[str, str]:
    bucket, separator, key = path.partition("/")
    if not separator or not bucket or not key:
        raise ParquetPlanningError("s3_path", f"invalid S3 path {path!r}")
    return bucket, key


def _new_s3_client(region: str) -> Any:
    import botocore.session

    session = botocore.session.get_session()
    if session.get_credentials() is None:
        raise ParquetPlanningError("source_identity", "ambient credentials vanished")
    return session.create_client("s3", region_name=region)


def _identity(
    descriptor: ScanDescriptor, path: str, s3_client: Any = None
) -> SourceIdentity:
    if descriptor.source_kind == "local":
        import pyarrow.fs as pafs

        info = descriptor.filesystem.get_file_info(path)
        if info.type != pafs.FileType.File:
            raise ParquetPlanningError("source_unavailable", f"missing {path!r}")
        if info.size is None or info.size < 0 or info.mtime_ns is None:
            raise ParquetPlanningError(
                "source_identity", f"no stable identity for {path!r}"
            )
        return SourceIdentity(int(info.size), f"mtime_ns:{int(info.mtime_ns)}")

    if s3_client is None:
        s3_client = _new_s3_client(str(descriptor.region))
    bucket, key = _split_s3(path)
    response = s3_client.head_object(Bucket=bucket, Key=key)
    size = response.get("ContentLength")
    if not isinstance(size, numbers.Integral) or isinstance(size, bool) or size < 0:
        raise ParquetPlanningError("source_identity", f"no stable size for {path!r}")
    version = response.get("VersionId")
    if version and version != "null":
        marker = f"version:{version}"
    else:
        etag, modified = response.get("ETag"), response.get("LastModified")
        if not etag or modified is None:
            raise ParquetPlanningError(
                "source_identity", f"no stable S3 marker for {path!r}"
            )
        timestamp = (
            modified.isoformat() if hasattr(modified, "isoformat") else str(modified)
        )
        marker = f"etag:{etag};last_modified:{timestamp}"
    return SourceIdentity(int(size), marker)


def read_footer(descriptor: ScanDescriptor) -> FooterSummary:
    import pyarrow.parquet as pq

    started = time.perf_counter()
    rows: list[RowGroup] = []
    footer_bytes = 0
    s3_client = (
        _new_s3_client(str(descriptor.region))
        if descriptor.source_kind == "s3"
        else None
    )
    try:
        for path, listed_size in zip(descriptor.paths, descriptor.listed_file_sizes):
            identity = _identity(descriptor, path, s3_client)
            if identity.size != listed_size:
                raise ParquetPlanningError(
                    "listed_size_changed", f"source changed: {path!r}"
                )
            with descriptor.filesystem.open_input_file(path) as source:
                metadata = pq.ParquetFile(source).metadata
            if _identity(descriptor, path, s3_client) != identity:
                raise ParquetPlanningError(
                    "source_changed", f"source changed while reading footer: {path!r}"
                )
            footer_bytes += int(getattr(metadata, "serialized_size", 0) or 0)

            column_indices: dict[str, list[int]] = {}
            for index in range(metadata.num_columns):
                name = metadata.schema.column(index).path.split(".", 1)[0]
                column_indices.setdefault(name, []).append(index)
            missing = [
                name for name in descriptor.projection if name not in column_indices
            ]
            if missing:
                raise ParquetPlanningError(
                    "missing_column", f"{path!r} lacks projected columns {missing!r}"
                )

            for row_group_id in range(metadata.num_row_groups):
                group = metadata.row_group(row_group_id)
                if group.num_rows == 0:
                    continue
                compressed = uncompressed = 0
                for column in descriptor.projection:
                    for index in column_indices[column]:
                        chunk = group.column(index)
                        csize, usize = (
                            chunk.total_compressed_size,
                            chunk.total_uncompressed_size,
                        )
                        if csize is None or csize < 0 or usize is None or usize < 0:
                            raise ParquetPlanningError(
                                "size_statistics",
                                f"row group sizes are unavailable for {path!r}",
                            )
                        compressed += int(csize)
                        uncompressed += int(usize)
                rows.append(
                    RowGroup(
                        path,
                        identity,
                        row_group_id,
                        int(group.num_rows),
                        compressed,
                        uncompressed,
                    )
                )
    finally:
        if s3_client is not None:
            with suppress(Exception):
                s3_client.close()
    rows.sort(key=lambda row: (row.path, row.row_group))
    return FooterSummary(tuple(rows), footer_bytes, time.perf_counter() - started)


def read_footer_with_retry(
    descriptor: ScanDescriptor, retried_io_errors: tuple[str, ...]
) -> FooterSummary:
    try:
        return next(
            iterate_with_retry(
                lambda: (read_footer(descriptor),),
                "read Parquet footer metadata for GPU planning",
                match=list(retried_io_errors),
            )
        )
    except ParquetPlanningError:
        raise
    except (OSError, IOError) as error:
        raise ParquetPlanningError("footer_io", str(error)) from error
    except Exception as error:
        # PyArrow and botocore expose several version-specific I/O exception
        # subclasses.  Treat those expected planning failures as capability
        # refusals while allowing plugin programming errors to remain visible.
        module = type(error).__module__
        if module.startswith(("pyarrow", "botocore", "boto3")):
            raise ParquetPlanningError("footer_io", str(error)) from error
        raise


def build_plan(footer: FooterSummary, target_tasks: int) -> ParquetPlan:
    if target_tasks <= 0:
        raise ParquetPlanningError("read_target", "target task count must be positive")
    if not footer.row_groups:
        return ParquetPlan(
            target_tasks,
            "parquet-read-v1:empty",
            (),
            footer.footer_bytes,
            footer.elapsed_s,
        )

    task_count = min(target_tasks, len(footer.row_groups))
    assignments: list[list[RowGroup]] = [[] for _ in range(task_count)]
    loads = [[0, 0, 0] for _ in range(task_count)]
    ordered = sorted(
        footer.row_groups,
        key=lambda row: (
            -row.uncompressed_bytes,
            -row.compressed_bytes,
            -row.num_rows,
            row.path,
            row.row_group,
        ),
    )
    for row in ordered:
        task_id = min(
            range(task_count),
            key=lambda i: (loads[i][0], loads[i][1], loads[i][2], i),
        )
        assignments[task_id].append(row)
        loads[task_id][0] += row.uncompressed_bytes
        loads[task_id][1] += row.compressed_bytes
        loads[task_id][2] += row.num_rows

    payload = [
        [
            [row.path, row.identity.size, row.identity.marker, row.row_group]
            for row in sorted(selected, key=lambda value: (value.path, value.row_group))
        ]
        for selected in assignments
    ]
    encoded = json.dumps(
        {"version": 1, "work": payload}, sort_keys=True, separators=(",", ":")
    ).encode()
    scheme_id = "parquet-read-v1:" + hashlib.sha256(encoded).hexdigest()

    work: list[ParquetWork] = []
    for work_id, selected in enumerate(assignments):
        files: dict[tuple[str, SourceIdentity], list[int]] = {}
        for row in selected:
            files.setdefault((row.path, row.identity), []).append(row.row_group)
        file_work = tuple(
            FileWork(path, identity, tuple(sorted(row_groups)))
            for (path, identity), row_groups in sorted(
                files.items(), key=lambda item: item[0][0]
            )
        )
        work.append(
            ParquetWork(
                work_id,
                scheme_id,
                file_work,
                sum(row.num_rows for row in selected),
                sum(row.compressed_bytes for row in selected),
                sum(row.uncompressed_bytes for row in selected),
            )
        )
    return ParquetPlan(
        target_tasks, scheme_id, tuple(work), footer.footer_bytes, footer.elapsed_s
    )


def serialize_work(work: ParquetWork) -> bytes:
    return pickle.dumps(work, protocol=5)


def deserialize_work(value: Any) -> ParquetWork:
    work = pickle.loads(bytes(value))
    if not isinstance(work, ParquetWork):
        raise TypeError("descriptor does not contain ParquetWork")
    return work


def descriptor_bundles(work: tuple[ParquetWork, ...]) -> list[RefBundle]:
    import pyarrow as pa

    bundles = []
    for descriptor in work:
        block = pa.table({"work": [serialize_work(descriptor)]})
        metadata = BlockAccessor.for_block(block).get_metadata()
        bundles.append(
            RefBundle(
                blocks=(BlockEntry(ray.put(block), metadata),),
                schema=block.schema,
                # These descriptor blocks are reconstructable InputDataBuffer
                # DAG roots. As with Ray's other reconstructable source inputs,
                # they are not lineage-owned output blocks for the executor's
                # block ref counter to destroy after their first consumer.
                owns_blocks=False,
            )
        )
    return bundles


@dataclass(frozen=True)
class ReadParquetConfig:
    source_kind: Literal["local", "s3"]
    region: Optional[str]
    projection: tuple[str, ...]
    retried_io_errors: tuple[str, ...]
    row_groups_per_read: int = 32

    def __post_init__(self) -> None:
        object.__setattr__(self, "projection", tuple(self.projection))
        object.__setattr__(self, "retried_io_errors", tuple(self.retried_io_errors))
        if self.source_kind not in ("local", "s3") or not self.projection:
            raise ValueError("invalid direct Parquet configuration")
        if self.source_kind == "s3" and not self.region:
            raise ValueError("S3 direct reads require a region")
        if self.row_groups_per_read <= 0:
            raise ValueError("row_groups_per_read must be positive")


class ReadParquetRuntime:
    def __init__(self, config: ReadParquetConfig, context: CudfRuntimeContext) -> None:
        self._config, self._context = config, context

    def initialize(self) -> None:
        self._context.ensure_cudf(
            source_kind=self._config.source_kind, region=self._config.region
        )

    def _runtime_identity(self, path: str) -> SourceIdentity:
        if self._config.source_kind == "local":
            import pyarrow.fs as pafs

            info = pafs.LocalFileSystem().get_file_info(path)
            if (
                info.type != pafs.FileType.File
                or info.size is None
                or info.size < 0
                or info.mtime_ns is None
            ):
                raise RuntimeError(f"Parquet source is unavailable: {path!r}")
            return SourceIdentity(int(info.size), f"mtime_ns:{int(info.mtime_ns)}")

        bucket, key = _split_s3(path)
        response = self._context.s3_client.head_object(Bucket=bucket, Key=key)
        size = response.get("ContentLength")
        if not isinstance(size, numbers.Integral) or isinstance(size, bool) or size < 0:
            raise RuntimeError(f"Parquet source has no stable size: {path!r}")
        version = response.get("VersionId")
        if version and version != "null":
            marker = f"version:{version}"
        else:
            etag, modified = response.get("ETag"), response.get("LastModified")
            if not etag or modified is None:
                raise RuntimeError(f"Parquet source has no stable marker: {path!r}")
            timestamp = (
                modified.isoformat()
                if hasattr(modified, "isoformat")
                else str(modified)
            )
            marker = f"etag:{etag};last_modified:{timestamp}"
        return SourceIdentity(int(size), marker)

    def _frames(self, files: tuple[FileWork, ...]):
        cudf = self._context.cudf
        for file in files:
            path = (
                "s3://" + urllib.parse.quote(file.path, safe="/")
                if self._config.source_kind == "s3"
                else file.path
            )
            for start in range(
                0, len(file.row_groups), self._config.row_groups_per_read
            ):
                kwargs: dict[str, Any] = {
                    "columns": list(self._config.projection),
                    "row_groups": list(
                        file.row_groups[
                            start : start + self._config.row_groups_per_read
                        ]
                    ),
                }
                if self._config.source_kind == "s3":
                    kwargs.update(
                        engine="cudf",
                        dataset_kwargs={"partitioning": None},
                        use_pandas_metadata=False,
                        categorical_partitions=False,
                    )

                def read_chunk():
                    if self._config.source_kind == "s3":
                        self._context.refresh_s3_for_task()
                    if self._runtime_identity(file.path) != file.identity:
                        raise RuntimeError(
                            f"Parquet source changed after planning: {file.path!r}"
                        )
                    frame = cudf.read_parquet(path, **kwargs)
                    if self._runtime_identity(file.path) != file.identity:
                        raise RuntimeError(
                            f"Parquet source changed during read: {file.path!r}"
                        )
                    return (frame,)

                frame = next(
                    iterate_with_retry(
                        read_chunk,
                        "read an exact Parquet row-group chunk with cuDF",
                        match=list(self._config.retried_io_errors),
                    )
                )
                if len(frame):
                    yield frame

    def apply(self, value: Any, task_context: Any) -> FrameStream:
        del task_context
        if not isinstance(value, ParquetWork):
            raise TypeError("read_parquet expected a ParquetWork descriptor")
        return FrameStream(value.work_id, self._frames(value.files))


__all__ = [
    "FooterSummary",
    "ParquetPlan",
    "ParquetPlanningError",
    "ParquetWork",
    "READ_PARQUET",
    "ReadParquetConfig",
    "ReadParquetRuntime",
    "ScanDescriptor",
    "ScanRecognition",
    "build_plan",
    "descriptor_bundles",
    "deserialize_work",
    "read_footer_with_retry",
    "recognize_scan",
    "serialize_work",
]
