#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>

#define LOOP_COUNT 100
#define M 20
#define P 200
#define N 20

int main()
{
    double *A, *B, *C;
    int i,j,r,max_threads,size;
    double alpha, beta;
    double s_initial, s_elapsed;
    
    printf("Intializing data for matrix multiplication C=A*B for matrix\n\n"
            " A(%i*%i) and matrix B(%i*%i)\n",M,P,P,N);
    alpha = 1.0;
    beta = 0.0;

    printf("Allocating memory for matrices aligned on 64-byte boundary for better performance \n\n");
    A = ( double *)mkl_malloc(M*P*sizeof( double ),64);
    B = ( double *)mkl_malloc(N*P*sizeof( double ),64);
    C = ( double *)mkl_malloc(M*N*sizeof( double ),64);
    if (A == NULL || B == NULL || C == NULL)
    {
        printf("Error: can`t allocate memory for matrices.\n\n");
        mkl_free(A);
        mkl_free(B);
        mkl_free(C);
        return 1;
    }

    printf("Intializing matrix data\n\n");
    size = M*P;
    for (i = 0; i < size; ++i)
    {
        A[i] = ( double )(i+1);
    }
    size = N*P;
    for (i = 0; i < size; ++i)
    {
        B[i] = ( double )(i-1);
    }

    printf("Finding max number of threads can use for parallel runs \n\n");
    max_threads = mkl_get_max_threads();

    printf("Running from 1 to %i threads \n\n",max_threads);
    for (i = 1; i <= max_threads; ++i)
    {
        size = M*N;
        for (j = 0; j < size; ++j)
        {
            C[j] = 0.0;
        }

	    printf("Requesting to use %i threads \n\n",i); 
	    mkl_set_num_threads(i);

	    printf("Measuring performance of matrix product using dgemm function\n"
		    " via CBLAS interface on %i threads \n\n",i);
	    s_initial = dsecnd();
	    for (r = 0; r < LOOP_COUNT; ++r)
	    {
    		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, P, alpha, A, P, B, N, beta, C, N);
            // multiply matrices with cblas_dgemm;
	    }
	    s_elapsed = (dsecnd() - s_initial) / LOOP_COUNT;

	    printf("Matrix multiplication using dgemm completed \n"
		    " at %.5f milliseconds using %d threads \n\n",
		    (s_elapsed * 1000),i);
        printf("Output the result: \n");
        size = M*N;
        for (i = 0; i < size; ++i)
        {
            printf("%i\t",(int)C[i]);
            if (i % N == N - 1)
                printf("\n");
        }
    }

    printf("Dellocating memory\n");
    mkl_free(A);
    mkl_free(B);
    mkl_free(C);

    return 0;
}
