.PHONY: bootstrap verify test unit integration gpu wheel smoke

PYTHON := .venv/bin/python

bootstrap:
	./scripts/bootstrap.sh

verify:
	$(PYTHON) scripts/verify_install.py

test:
	./scripts/test.sh

unit:
	$(PYTHON) -m pytest -q plugin/tests/unit plugin/tests/compatibility

integration:
	$(PYTHON) -m pytest -q -m "not gpu and not s3" plugin/tests/integration

gpu:
	CUDA_VISIBLE_DEVICES=$${CUDA_VISIBLE_DEVICES:-0} $(PYTHON) -m pytest -q -m gpu plugin/tests

wheel:
	$(PYTHON) scripts/build_patched_ray_wheel.py
	$(PYTHON) -m pip wheel --no-deps --no-build-isolation --wheel-dir dist ./plugin

smoke:
	$(PYTHON) benchmark/smoke.py
