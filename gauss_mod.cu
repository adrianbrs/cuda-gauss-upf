/***************************
 * Adrian Cerbaro - 178304 *
 ***************************/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <assert.h>

/**
 * Habilita logs de depuração se estiver definido.
 */
// #define DEBUG

/**
 * Define uma macro `println` que adiciona `\n` ao final do printf.
 * Quando chamada a partir do device, também adiciona os índices do bloco e thread.
 */
#ifdef __CUDA_ARCH__
// fprintf não é suportado no device, mas é preciso definir aqui para que o símbolo seja encontrado na compilação
#define fprintln(stream, fmt, args...) fprintf(stream, fmt "\n", ##args)
#define println(fmt, args...) printf("\e[2m[(%d,%d,%d) (%d,%d,%d)]\e[0m " fmt "\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, ##args)
#define errorln(fmt, args...) println(fmt, ##args)
#define exiterr(fmt, args...) errorln(fmt, ##args); asm("exit")
#else
#define fprintln(stream, fmt, args...) fprintf(stream, fmt "\n", ##args)
#define println(fmt, args...) fprintln(stdout, fmt, ##args)
#define errorln(fmt, args...) fprintln(stderr, fmt, ##args)
#define exiterr(fmt, args...) errorln(fmt, ##args); exit(EXIT_FAILURE)
#endif

#ifdef DEBUG
#define debug(fmt, args...) println(fmt, ##args)
#else
#define debug(fmt, args...)
#endif

#define GB_IN_BYTES (1024 * 1024 * 1024)

enum ManagementMode {
	Host,
	Device
};

struct KernelIndexProps {
	int blocks;
	int threads;
	int block_idx;
	int thread_idx;
	int idx;
};

// Alterar entre Host e Device para testar eficiência
ManagementMode BACK_SUBSTITUTION_MODE = Host;

int BLOCKS = -1;
int THREADS = -1;

void saveResult(double *A, double *b, double *x, int n);
int  testLinearSystem(double *A, double *b, double *x, int n);
void loadLinearSystem(int n, double *A, double *b);
void solveLinearSystem(const double *A, const double *b, double *x, int n);
void computeBlocksAndThreads(int *blocks, int *threads, cudaDeviceProp *props, int n);

__global__ void kernel_GaussianElimination(double *A, double *b, int i, int n);
__global__ void kernel__BackSubstitution(double *A, double *b, double *x, int n);
__device__ KernelIndexProps device_getKernelIndexProps();

inline void ensureSuccess(cudaError_t op_result);

/**
 * Função utilitária para obter o modo de gerenciamento (host ou device) a partir de uma variável de ambiente.
 */
inline ManagementMode getManagementModeEnv(const char *env_name, const ManagementMode default_mode) {
	char *mode = getenv(env_name);
	if (mode == NULL) return default_mode;
	if (strcasecmp(mode, "host") == 0) return Host;
	if (strcasecmp(mode, "device") == 0) return Device;
	exiterr("Modo \"%s\" inválido (deve ser \"host\" ou \"device\")", mode);
	return (ManagementMode) -1;
}

int main(int argc, char **argv) {
	char *name = argv[0];
	if (argc < 2) {
		exiterr("Uso: %s <n>", name);
	}

	// Carrega o modo de back-substitution
	BACK_SUBSTITUTION_MODE = getManagementModeEnv("BACK_SUBSTITUTION_MODE", BACK_SUBSTITUTION_MODE);
	
	// Carrega configurações fixas de blocos e threads
	if (getenv("BLOCKS")) BLOCKS = atoi(getenv("BLOCKS"));
	if (getenv("THREADS")) THREADS = atoi(getenv("THREADS"));

	println();
	println("+----- Variáveis de Ambiente ------+");
	println("BACK_SUBSTITUTION_MODE: %s", BACK_SUBSTITUTION_MODE == Host ? "Host" : "Device");
	println("BLOCKS: %d", BLOCKS);
	println("THREADS: %d", THREADS);
	println("+---------------------------------+");
	println();
	
	int n = atoi(argv[1]);
	int nerros = 0;

	ensureSuccess(cudaSetDevice(0));

	double *A = (double *) malloc(n * n * sizeof(double));
	double *b = (double *) malloc(n * sizeof(double));
	double *x = (double *) malloc(n * sizeof(double));  

    loadLinearSystem(n, &A[0], &b[0]);
      
    solveLinearSystem(&A[0], &b[0], &x[0], n);

	nerros += testLinearSystem(&A[0], &b[0], &x[0], n);

	println("Errors=%d", nerros);
    
    saveResult(&A[0], &b[0], &x[0], n);
    
	return EXIT_SUCCESS;
}

int runMain(int argc, char **argv) {
	return main(argc, argv);
}

void saveResult(double *A, double *b, double *x, int n) {	
	int i;
    FILE *res;	
    
    res = fopen("result.out", "w");
	if (res == NULL){
	    println("File result.out does not open");
	    exit(1);
	}
	
	for(i=0; i < n; i++){
	    fprintln(res, "%.6f", x[i]);
    }
	
	fclose( res );
}

int testLinearSystem(double *A, double *b, double *x, int n) {
	int i, j, c =0;
	double sum = 0;

	for (i = 0; i < n; i++) {
		sum=0;
		for (j = 0; j < n; j++)
			sum += A[i * n + j] * x[j];		
		if (abs(sum - b[i]) >= 0.001) {
		    // println("%f", (sum - b[i]) );
			c++;
		}
	}
	return c;
}

void loadLinearSystem(int n, double *A, double *b) {
	int i, j;
	FILE *mat, *vet;
		
        mat = fopen("matrix.in", "r");
	if (mat == NULL){
	    println("File matrix.in does not open");
	    exit(1);
	}
	
	vet = fopen("vector.in", "r");
	if (vet == NULL){
	    println("File vector.in does not open");
	    exit(1);
	}	

	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++)
			fscanf(mat, "%lf", &A[i * n + j]);
	}

	for (i = 0; i < n; i++)
		fscanf(vet, "%lf", &b[i]);
		
	fclose ( mat );
	fclose ( vet );			
}

void solveLinearSystem(const double *A, const double *b, double *x, int n) {
	int devid, clockRateKHz;
	struct cudaDeviceProp props;

	ensureSuccess(cudaGetDevice(&devid));
    ensureSuccess(cudaGetDeviceProperties(&props, devid));
	ensureSuccess(cudaDeviceGetAttribute(&clockRateKHz, cudaDevAttrClockRate, devid));
	
	// Calcula quantidade de blocos e threads
	int blocks, threads;
	computeBlocksAndThreads(&blocks, &threads, &props, n);

	cudaFuncAttributes attr;
	cudaFuncGetAttributes(&attr, kernel_GaussianElimination);

	println("+------- Informações Gerais -------+");
	println("Nome da GPU: %s", props.name);
	println("Blocos: %d", blocks);
	println("Threads: %d", threads);
	println("Threads por bloco: %d", props.maxThreadsPerBlock);
	println("Threads por SM: %d", props.maxThreadsPerMultiProcessor);
	println("Blocos por SM: %d", props.maxBlocksPerMultiProcessor);
	println("SMs: %d", props.multiProcessorCount);
	println("Clock: %d MHz", clockRateKHz / 1000);
	println("Warp Size: %d", props.warpSize);
	println("Registradores usados: %d", attr.numRegs);
	println("Memória Local (Spill) usada: %ld bytes", attr.localSizeBytes);
	println("Registradores por Bloco: %d", props.regsPerBlock);
	println("Registradores por SM: %d", props.regsPerMultiprocessor);
	println("+---------------------------------+");
	println();

	// Medição simples do tempo total
	struct timespec tstart, tend;
	clock_gettime(CLOCK_MONOTONIC, &tstart);

	size_t Asize = n * n * sizeof(double);
	size_t bsize = n * sizeof(double);
	size_t xsize = n * sizeof(double);
	double *A_d, *b_d, *x_d;

	ensureSuccess(cudaMalloc(&A_d, Asize));
	ensureSuccess(cudaMalloc(&b_d, bsize));
	if (BACK_SUBSTITUTION_MODE == Device) {
		ensureSuccess(cudaMalloc(&x_d, xsize));
	}

	ensureSuccess(cudaMemcpy(A_d, A, Asize, cudaMemcpyHostToDevice));
	ensureSuccess(cudaMemcpy(b_d, b, bsize, cudaMemcpyHostToDevice));

	// Eliminação Gaussiana
	for (int i = 0; i < (n - 1); ++i) {
		// Recalcula os blocos a cada pivô para evitar lançar um kernel muito maior
		// do que o necessário nas iterações finais
		blocks = min(blocks, n - i - 1);
		if (blocks < 1) blocks = 1;
		kernel_GaussianElimination<<<blocks, threads>>>(A_d, b_d, i, n);
	}

	// Back-substitution no device
	if (BACK_SUBSTITUTION_MODE == Device) {
		kernel__BackSubstitution<<<1, 1>>>(A_d, b_d, x_d, n);
	}

	// Checar erros de lançamento antes de bloquear na cópia
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		errorln("Erro no lançamento do kernel: %s\n", cudaGetErrorString(error));
	}

	if (BACK_SUBSTITUTION_MODE == Device) {
		// Copia bloqueante para o host
		ensureSuccess(cudaMemcpy(x, x_d, xsize, cudaMemcpyDeviceToHost));
		cudaFree(x_d);
	} else {
		// Back-substitution no host
		double *Acpy, *bcpy;
		Acpy = (double *) malloc(Asize);
		bcpy = (double *) malloc(bsize);
		ensureSuccess(cudaMemcpy(Acpy, A_d, Asize, cudaMemcpyDeviceToHost));
		ensureSuccess(cudaMemcpy(bcpy, b_d, bsize, cudaMemcpyDeviceToHost));

		x[n - 1] = bcpy[n - 1] / Acpy[(n - 1) * n + n - 1];
		for (int i = (n - 2); i >= 0; i--) {
			double temp = bcpy[i];
			for (int j = (i + 1); j < n; j++) {
				temp -= (Acpy[i * n + j] * x[j]);
			}
			x[i] = temp / Acpy[i * n + i];
		}

		free(Acpy);
		free(bcpy);
	}

	// Calcula o tempo total
	clock_gettime(CLOCK_MONOTONIC, &tend);
	double elapsed = (tend.tv_sec - tstart.tv_sec) + (tend.tv_nsec - tstart.tv_nsec) / 1e9;

	println("+--------- Resultados -------------+");
	println("+ Tempo total: %f s", elapsed);
	println("+----------------------------------+");
	println();

	cudaFree(A_d);
	cudaFree(b_d);
}

/**
 * Calcula a quantidade de blocos e threads necessárias para resolver uma matrix de `n*n`.
 */
inline void computeBlocksAndThreads(int *blocks, int *threads, cudaDeviceProp *props, int n) {
	if (BLOCKS > 0) *blocks = BLOCKS;
	else  *blocks = props->multiProcessorCount * 32;
	if (*blocks > props->maxGridSize[0]) *blocks = props->maxGridSize[0];

	if (THREADS > 0) *threads = THREADS;
	else *threads = 512;
	if (*threads > props->maxThreadsPerBlock) *threads = props->maxThreadsPerBlock;
	else if (*threads > n) *threads = n;
	else if (*threads <= 0) *threads = 1;
}

/**
 * Kernel para etapa de Eliminação Gaussiana.
 * 
 * @param A Matriz de coeficientes
 * @param b Vetor de termos independentes
 * @param i Índice da linha do pivô
 * @param n Tamanho da matriz (`n x n`)
 */
__global__ void kernel_GaussianElimination(double *A, double *b, int i, int n) {
	KernelIndexProps props = device_getKernelIndexProps();
	bool active = true;

	// thread atual já é maior que a quantidade de elementos a serem processados por linha
	// ou bloco atual já é maior que a quantidade de linhas a serem processadas
	if (props.thread_idx >= n || props.block_idx >= n) {
		debug("Número de threads ou blocos maior que N, isso pode resultar em erros inesperados.");
		active = false;
	}

	// l_idx       = índice da linha sendo processada abaixo do pivô
	// l_idx_min   = índice da linha após o pivô (i + 1)
	// l_idx_start = índice da primeira linha a ser processada pelo bloco
	// l_idx_jumps = quantos pulos de `blocks` faltam para chegar em `l_idx_min`
	// e_idx       = índice do elemento da linha sendo processada
	// r           = rodada atual
	int l_idx, l_idx_min, l_idx_start, l_idx_jumps, e_idx, r;
	// razão do elemento logo abaixo do pivô com o pivô
	double ratio;
	// quantidade de elementos a serem processados na linha atual
	// será sempre `n-i` (quantidade de elementos do pivô em diante)
	int rem_n = n - i;
	// cada thread deverá processar um elemento da linha, então precisamos calcular
	// quantas rodadas são necessárias quando n > threads
	int rounds = rem_n > props.threads ? (rem_n + props.threads - 1) / props.threads : 1;
	// elemento da diagonal principal na linha `i` (pivô)
	double pivot = A[i * n + i];
	
	l_idx_min = i + 1;
	l_idx_start = props.block_idx;
	if (l_idx_start < l_idx_min) {
		l_idx_jumps = (l_idx_min - l_idx_start + props.blocks - 1) / props.blocks;
		l_idx_start += l_idx_jumps * props.blocks;
	}

	if (props.thread_idx == 0) {
		debug(">>> i=%d, rounds=%d, rem_n=%d, l_idx_start=%d, blocks=%d", i, rounds, rem_n, l_idx_start, props.blocks);
	}

	for (l_idx = l_idx_start; l_idx < n; l_idx += props.blocks) {
		ratio = A[l_idx * n + i] / pivot;
		__syncthreads();

		// Evita "bagunçar" com os dados caso a quantidade de blocos/threads seja alocada de forma incorreta
		if (!active) continue;

		debug("i=%d, l_idx=%d", i, l_idx);

		for (r = 0; r < rounds; ++r) {
			// inicia sempre no mesmo índice do pivô
			e_idx = i + (r * props.threads) + props.thread_idx;
			if (e_idx >= n) break;
			A[l_idx * n + e_idx] -= (ratio * A[i * n + e_idx]);
		}

		// Deve ser executado uma única fez por linha
		if (props.thread_idx == 0) {
			b[l_idx] -= (ratio * b[i]);
		}
	}
}

/**
 * Kernel para etapa de Back-substitution.
 *
 * Manter o Back-Substitution no device, mesmo que de forma sequencial, evita copiar toda a matriz (*A) e o vetor (*b) para o host.
 * 
 * É vantajoso?
 */
__global__ void kernel__BackSubstitution(double *A, double *b, double *x, int n) {
	KernelIndexProps props = device_getKernelIndexProps();
	// processa somente na primeira thread do primeiro bloco
	if (props.idx == 0) {
		int i, j;

		/* Back-substitution */
		x[n - 1] = b[n - 1] / A[(n - 1) * n + n - 1];
		for (i = (n - 2); i >= 0; i--) {
			double temp = b[i];
			for (j = (i + 1); j < n; j++) {
				temp -= (A[i * n + j] * x[j]);
			}
			x[i] = temp / A[i * n + i];
		}
	}
}

/**
 * Retorna índices sequenciais para a thread e o bloco, como se ambos fossem vetores e não matrizes.
 */
__device__ KernelIndexProps device_getKernelIndexProps() {
	// total de blocos
    int blocks = gridDim.x * gridDim.y * gridDim.z;
	// total de threads por bloco
    int threads = blockDim.x * blockDim.y * blockDim.z;
	// índice sequencial do bloco no grid (0 -> blocks - 1)
	int block_idx = (blockIdx.y * gridDim.x) + (blockIdx.z * gridDim.x * gridDim.y) + blockIdx.x;
	// índice sequencial da thread no bloco (0 -> threads - 1)
    int thread_idx = (threadIdx.y * blockDim.x) + (threadIdx.z * blockDim.x * blockDim.y) + threadIdx.x;
	// índice sequencial da thread no grid (0 -> blocks * threads - 1)
	int idx = (block_idx * threads) + thread_idx;
	KernelIndexProps props = {
		blocks,
		threads,
		block_idx,
		thread_idx,
		idx
	};
	return props;
}

/**
 * Função utilitária para encerrar o programa caso o resultado de uma operação CUDA falhe no host.
 */
inline void ensureSuccess(cudaError_t op_result) {
    if (op_result != cudaSuccess) {
        errorln("Operação CUDA falhou: %d\n", op_result);
        exit(EXIT_FAILURE);
    }
}