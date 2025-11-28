.PHONY: all clean build run
.DEFAULT_GOAL:=all

n?=7500
bsmode?=host
bmulti?=4
tfactor?=9
times?=5

all: clean build run
seq: clean build-seq run-seq

env:
	test -d .venv || python -m venv .venv && .venv/bin/python -m pip install --upgrade pip wheel setuptools

freeze:
	.venv/bin/python -m pip freeze > requirements.txt

install: env
	.venv/bin/python -m pip install -r requirements.txt

build:
	nvcc ./gauss_mod.cu -o gauss_mod_cuda

run:
	BACK_SUBSTITUTION_MODE=$(bsmode) BLOCKS_MULTIPLIER=$(bmulti) THREADS_FACTOR=$(tfactor) ./gauss_mod_cuda $(n)

clean:
	rm -f ./gauss_mod_cuda ./gauss_mod

build-seq:
	gcc ./gauss_mod.c -o gauss_mod

run-seq:
	echo "$(n)" | ./gauss_mod

benchmark:
	./benchmark.sh $(n) $(bsmode) $(bmulti) $(tfactor) $(times)

benchmark-seq:
	./benchmark_seq.sh $(n) $(times)

benchmark-clean:
	rm -f ./benchmarks/*.log