#include <stdio.h>
#include <mkl.h>
#define M 10
#define N 10
#define P 10

int main()
{
    int i,j;
    double *A, *B;
    double alpha, beta;
    alpha = 1.0;
    beta = 0.0;

    A = (double*)mkl_malloc( M * N * sizeof(double),64);
    B = (double*)mkl_malloc(P * sizeof(double),64);
    
    if (A == NULL || B == NULL)
    {
        printf("Can not allocate memory\n");
        mkl_free(A);
        mkl_free(B);
        return 1;
    }
    
    int size = M * N;
    for (i = 0; i < size; ++i)
    {
        A[i] = i / 2;
    }
    for (j = 0; j < P; ++j)
    {
        /*B[j] = j / 2;*/
        B[j] = j * 2; // the incr should be integer
    }

    double *C;
    C = (double*)mkl_malloc(  P * sizeof(double), 64);
    if (C == NULL)
    {
        mkl_free(C);
        return 1;
    }
    for (i = 0; i < P; ++i)
    {
        C[i] = i;
    }

    cblas_dgemv(CblasRowMajor, CblasNoTrans, M, N, alpha, A, M, B, 2, beta, C, 1);
    
    size = P;
    for (i = 0; i < size; ++i)
    {
        printf("%d\t",C[i]);
        if (i % 5 == 4)
        {
            printf("\n");
        }
    }

    mkl_free(A);
    mkl_free(B);
    mkl_free(C);
    return 0;
}

