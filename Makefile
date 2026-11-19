# Every target below runs on CPU only.
#
# .RECIPEPREFIX is set so that recipes use ">" instead of a tab; GNU make 3.82+
# accepts this and it keeps the file robust across editors.
.RECIPEPREFIX = >

PY ?= python3
OUT ?= results/recomputed

.PHONY: all tables verify test clean help

help:
> @echo "make tables  - rebuild every paper table from the shipped records"
> @echo "make verify  - rebuild and compare against the published values"
> @echo "make test    - unit tests for dose, bootstrap duplication and hooks"
> @echo "make all     - tables + verify + test"
> @echo "make clean   - remove results/recomputed"

all: tables verify test

tables:
> $(PY) scripts/reproduce_tables.py --out $(OUT)

verify:
> $(PY) scripts/verify_release.py --out $(OUT)

test:
> $(PY) -m pytest -q tests

clean:
> rm -rf $(OUT)
