#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#define  Max(a,b) ((a)>(b)?(a):(b))

double   maxeps = 0.1e-7;
int itmax = 100;
int i,j,k;
double eps;
double *A;
double *B;
int N, NN;

void relax();
void resid();
void init();
void verify();

int main(int an, char **as)
{
	int it;
	int m[] = {50, 100, 150};
	for (int l = 0; l < 3; l++)
	{
	N = m[l];
	NN = N * N;
	printf("SIZE = %d;\n", m[l]);
	int max_threads_count = omp_get_max_threads();
	for(int threads = 1; threads < max_threads_count + 1; threads *= 2)
	{
	printf("threads = %i; ", threads);
	omp_set_num_threads(threads);
	double start = omp_get_wtime();
	init();
	for(it=1; it<=itmax; it++)
	{
		eps = 0.;
		relax();
		resid();
		if (eps < maxeps) break;
	}
	verify();
	double end = omp_get_wtime();
	printf("TIME: %f\n", end - start);
	}
	free(A);
	free(B);
	}
	return 0;
}

void init()
{

	A = malloc(N * NN * sizeof(double));
	B = malloc(N * NN * sizeof(double));
	for(i=0; i<=N-1; i++)
	for(j=0; j<=N-1; j++)
	for(k=0; k<=N-1; k++)
	{
		if(i==0 || i==N-1 || j==0 || j==N-1 || k==0 || k==N-1)
			A[i*NN + j*N + k] = 0.;
		else
			A[i*NN + j*N + k] = (4. + i + j + k) ;
	}
}

void relax()
{
#pragma omp parallel for shared (A, B, N, NN) private(i,j,k)

	for(i=1; i<=N-2; i++)
	for(j=1; j<=N-2; j++)
	for(k=1; k<=N-2; k++)

	{
		B[i*NN + j*N + k] = (A[(i-1)*NN + j*N + k] + A[(i+1)*NN + j*N + k] + A[i*NN + (j-1)*N + k]+A[i*NN + (j+1)*N + k] + A[i*NN + j*N + (k-1)] + A[i*NN + j*N + (k+1)])/6.;
	}
}

void resid()
{
#pragma omp parallel for shared (A, B, N, NN) private(i,j,k) reduction(max:eps)

	for(i=1; i<=N-2; i++)
	for(j=1; j<=N-2; j++)
	for(k=1; k<=N-2; k++)
	{
		double e = fabs(A[i*NN + j*N + k] - B[i*NN + j*N + k]);
		A[i*NN + j*N + k] = B[i*NN + j*N + k];
		eps = Max(eps,e);
	}
}

void verify()
{
	double s = 0.;
#pragma omp parallel for shared(A, N, NN) private(i,j,k) reduction(+:s)
	for(i=0; i<=N-1; i++)
	for(j=0; j<=N-1; j++)
	for(k=0; k<=N-1; k++)
	{
		s += A[i*NN + j*N + k] * (i + 1) * (j + 1) * (k + 1)/(N * N * N);
	}
	printf("S = %f; ",s);
}
