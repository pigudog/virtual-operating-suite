PY ?= python
DEVICE ?= cuda

.PHONY: smoke
smoke:
	$(PY) examples/run_suite_example.py --device $(DEVICE)
