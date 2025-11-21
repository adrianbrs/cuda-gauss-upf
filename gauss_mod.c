#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <assert.h>

void showMatrix(int n, double *A);
void saveFiles(int n, double *A, double *b);
void saveResult(double *A, double *b, double *x, int n);
int  testLinearSystem(double *A, double *b, double *x, int n);
void generateLinearSystem(int n, double *A, double *b);
void loadLinearSystem(int n, double *A, double *b);
void solveLinearSystem(const double *A, const double *b, double *x, int n);

int main(int argc, char **argv) {
	int n;
	scanf("%d", &n);
	
	int nerros = 0;

	double *A = (double *) malloc(n * n * sizeof(double));
	double *b = (double *) malloc(n * sizeof(double));
	double *x = (double *) malloc(n * sizeof(double));

    //generateLinearSystem(n, &A[0], &b[0]);
    //saveFiles(n, &A[0], &b[0]);      

    loadLinearSystem(n, &A[0], &b[0]);
      
    solveLinearSystem(&A[0], &b[0], &x[0], n);

	nerros += testLinearSystem(&A[0], &b[0], &x[0], n);

	printf("Errors=%d\n", nerros);
    
    saveResult(&A[0], &b[0], &x[0], n);
    
	return EXIT_SUCCESS;
}

void showMatrix(int n, double *A){
    int i, j;
    
    for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++)
		    printf("%.6f\t", A[i * n + j]);
		printf("\n");
	}
}

void saveFiles(int n, double *A, double *b) {
	int i, j;	
    FILE *mat, *vet;
		
    mat = fopen("matrix.in", "w");
	if (mat == NULL){
	    printf("File matrix.in does not open\n");
	    exit(1);
	}
	
	vet = fopen("vector.in", "w");
	if (vet == NULL){
	    printf("File vector.in does not open\n");
	    exit(1);
	}	
			
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++)
		    fprintf(mat, "%.6f\t", A[i * n + j]);
		fprintf(mat, "\n");

	    fprintf(vet, "%.6f\n", b[i]);
	}
	
	fclose ( mat );
	fclose ( vet );
}

void saveResult(double *A, double *b, double *x, int n) {	
	int i;
    FILE *res;	
    
    res = fopen("result.out", "w");
	if (res == NULL){
	    printf("File result.out does not open\n");
	    exit(1);
	}
	
	for(i=0; i < n; i++){
	    fprintf(res, "%.6f\n", x[i]);
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
		    printf("%f\n", (sum - b[i]) );
			c++;
		}
	}
	return c;
}

void generateLinearSystem(int n, double *A, double *b) {
	int i, j;
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++)
			A[i * n + j] = (1.0 * n + (rand() % n)) / (i + j + 1);
		A[i * n + i] = (10.0 * n) / (i + i + 1);
	}
    
	for (i = 0; i < n; i++)
		b[i] = 1.;		
}

void loadLinearSystem(int n, double *A, double *b) {
	int i, j;	
	FILE *mat, *vet;
		
        mat = fopen("matrix.in", "r");
	if (mat == NULL){
	    printf("File matrix.in does not open\n");
	    exit(1);
	}
	
	vet = fopen("vector.in", "r");
	if (vet == NULL){
	    printf("File vector.in does not open\n");
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
	double *Acpy = (double *) malloc(n * n * sizeof(double));
	double *bcpy = (double *) malloc(n * sizeof(double));
	memcpy(Acpy, A, n * n * sizeof(double));
	memcpy(bcpy, b, n * sizeof(double));

	int i, j, count;

	/* Gaussian Elimination */
	for (i = 0; i < (n - 1); i++) {

		for (j = (i + 1); j < n; j++) {
			double ratio = Acpy[j * n + i] / Acpy[i * n + i];
			for (count = i; count < n; count++) {
				Acpy[j * n + count] -= (ratio * Acpy[i * n + count]);
			}
			bcpy[j] -= (ratio * bcpy[i]);
		}
	}

	/* Back-substitution */
	x[n - 1] = bcpy[n - 1] / Acpy[(n - 1) * n + n - 1];
	for (i = (n - 2); i >= 0; i--) {
		double temp = bcpy[i];
		for (j = (i + 1); j < n; j++) {
			temp -= (Acpy[i * n + j] * x[j]);
		}
		x[i] = temp / Acpy[i * n + i];
	}
}

