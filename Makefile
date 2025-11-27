.PHONY: all clean build run
.DEFAULT_GOAL:=all

n?=7500
bsmode?=host
times?=5

all: clean build run
seq: clean build-seq run-seq

build:
	nvcc ./gauss_mod.cu -o gauss_mod_cuda

run:
	BACK_SUBSTITUTION_MODE=$(bsmode) ./gauss_mod_cuda $(n)

clean:
	rm -f ./gauss_mod_cuda ./gauss_mod

build-seq:
	gcc ./gauss_mod.c -o gauss_mod

run-seq:
	echo "$(n)" | ./gauss_mod

benchmark:
	./benchmark.sh $(n) $(bsmode) $(times)

benchmark-seq:
	./benchmark_seq.sh $(n) $(times)

benchmark-clean:
	rm -rf ./benchmarks/