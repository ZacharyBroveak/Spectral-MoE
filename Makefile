# Spectral-MoE: Makefile
# Author: Zachary Broveak
#
# Command-line entry points; a thin wrapper around python -m src.pipeline (see
# src/pipeline.py).
#
# Targets: help, prepare, train ARM=<name>, run, evaluate, report. make help lists every ARM
# name.
#
# Variables:
# - PYTHON: the interpreter with requirements.txt installed.
# - OUT: the output directory, results by default (data, runs, evaluation, report). Keep one
#   OUT per configuration.
# - GPUS: 1 or 4. With 1, learned training processes its four data shards in sequence.
# - SESSION_SECONDS: pause after this many seconds at the next saved boundary (0 disables it);
#   rerun the same command to resume. Python exits with status 75 on a pause, which Make
#   reports as error 2.
#
# PYTHONDONTWRITEBYTECODE keeps __pycache__ directories out of the source tree.

PYTHON ?= python
OUT ?= results
GPUS ?= 1
SESSION_SECONDS ?= 0
ARM ?=
export PYTHONDONTWRITEBYTECODE = 1
PIPE = $(PYTHON) -m src.pipeline
ARGS = --out "$(OUT)" --gpus $(GPUS) --session-seconds $(SESSION_SECONDS)

.PHONY: help prepare train run evaluate report
help:
	@echo "make prepare | train ARM=<name> | run | evaluate | report"
	@echo "Variables: GPUS=1|4 OUT=results SESSION_SECONDS=0 PYTHON=python"
	@echo "ARM names:"
	@$(PYTHON) -c "from src.training.recipes import RECIPES; print(*sorted(RECIPES), sep=chr(10))"
prepare run evaluate report:
	@$(PIPE) $@ $(ARGS)
train:
	@$(PIPE) train $(ARGS) $(if $(ARM),--arm "$(ARM)",)
