.PHONY: test compile audit help

help:
	@echo "make test      run unit tests"
	@echo "make compile   compile source and scripts"
	@echo "make audit     audit the two formal result bundles"

test:
	python -m pytest -q

compile:
	python -m compileall -q src scripts

audit:
	python -m epilepsy_eeg.cli audit --task1-root results/task1_annotation_detection --task2-root results/task2_morphology_pg36
