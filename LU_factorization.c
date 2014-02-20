/*
 * warning! This program using int instead of size_t as the size of matrix.
 * It will save your time in malloc memory without any impact when the matrix is not huge.
 * But if the matrix is too huge (for example, more than 30000 rows and 30000 cols), this wil cause trouble.
 * Of course, you can change the ints to size_ts if needed.
 */
#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>

void Incopy(int sign,float *change,float *unchange,int out,int in,int index,int n);

void Init(int N,float *A);

void CorrectOrNot(float *Array,int NN);

int main(int argc, char *argv[])
{
    // start time record	
	double start = dsecnd();

    if (argc < 2)
    {
        return 1;
    }
	const int m = atoi(argv[1]);
    if (m <= 0)
    {
        return 1;
    }
	const int n = m;
	const int nb = 256;
	
    float *c = (float *)( mkl_malloc((m*n) * sizeof(float),64));
    if (c == NULL)
    {
        printf("LU factorization failed. The matrix is too huge!\n");
        mkl_free(c);
        return 0;
        
    }
	// initialize matrix c
	Init(m,c);


    float *temp3 = ( float *)( mkl_malloc((m*n) * sizeof(float),64));
    if (temp3 == NULL)
    {
        printf("LU factorization failed.  The matrix is too huge!\n");
        mkl_free(c);
        return 0;
    }
	int i,i1,i2,j,k;

    // case 1, the matrix is small enough
	if((m <= nb)||(n <= nb))
	{
		int b[ m ];
		int info = LAPACKE_sgetrf(CblasRowMajor,m,n,c,n,b);
		if(info < 0)
			{
				printf("LU factorization failed\n");
                printf("info = %d\n",info);
				mkl_free(c);
				return 0;
			}
		else
		// if LU factorization succeed	
		{
			for(i1 = 0,j = 0;j < m;j++)
			{
				for(k = 0;k < n;k++)
                    printf("%f ",c[i1++]);
                printf("\n");
			}
		}
	}
	// LU factorization for huge matrix - divide it into blocks
	else
    {	
	    for(i = 0,i2 = 0;i2 < (m-nb);i++,i2 += nb)
	    {
		    //1:matrix temp1 for the LU factorization in the left top small block
		
            float *temp1 = ( float *)( mkl_malloc((n*nb) * sizeof(float),64));
		    Incopy(0,temp1,c,nb,(n - i * nb),( n * i + i) * nb,n);
		

		    //use LAPACKE_sgetrf() for the LU factorization
		    int mm = nb;
		    int nn = (n - i * nb);
		    int lda = nn;
		    int b[m];
		    int info = LAPACKE_sgetrf(101,mm,nn,temp1,lda,b);

		    if(info < 0)
			    {
                    printf("LU factorization failed\n");
                    printf("info = %d\n",info);
				    return 0;
			    }

		    else
		    // store the result to the output matrix
		    {
			
			    Incopy(1,c,temp1,nb,(n - i * nb),(n * i + i)*nb,n);

		    }
		
		    // maintain the left part of matrix temp1 for further calculation
		
		    Incopy(0,temp1,c,nb,nb,(n * i + i)*nb,n);
		

		    //2:construct matrix 2 
	        float *temp2 = (float *)( mkl_malloc((m*nb) * sizeof(float),64));
		
		    Incopy(0,temp2,c,(m -(i + 1)* nb),nb,n * nb * (i + 1) + i * nb,n);
		

		    // use cblas_strsm() for matrix 2
		    const  CBLAS_ORDER ORDER = CblasRowMajor;
		    const  CBLAS_SIDE SIDE = CblasRight;
		    const  CBLAS_UPLO UPLO = CblasUpper;
		    const  CBLAS_TRANSPOSE TRANSA = CblasNoTrans;
		    const  CBLAS_DIAG DIAG = CblasNonUnit;
		    mm = ( m - ( i+1 )*nb);
		    nn = nb;
		    cblas_strsm(ORDER,SIDE,UPLO,TRANSA,DIAG,mm,nn,1,temp1,nb,temp2,nb);

		    //store matrix 2
		
		    Incopy(1,c,temp2,( m - (i+1) *nb),nb,n * nb *(i+1)+i*nb,n);
		
		    //maintain the right part of matrix temp1 for further calculation
		
		    Incopy(0,temp1,c,nb,(n -i *nb -nb),(n*i + i)*nb + nb,n);
		
		    //3:construct temp3
		
		    Incopy(0,temp3,c,(m -(i + 1)*nb),(n -(i + 1)*nb),n *nb *(i+1)+i *nb + nb,n);
		
		    // use cblas_sgemm() for calculate temp3
		    CBLAS_ORDER cblasRowmajor = CblasRowMajor;
		    const  CBLAS_TRANSPOSE TransA = CblasNoTrans;
		    const  CBLAS_TRANSPOSE TransB = CblasNoTrans;
		    mm = (m-(i+1)*nb);
		    nn = (n-(i+1)*nb);
		    int kk = nb;
		    lda = kk;
		    int ldb = nn;
		    int ldc = nn;
		    const float alpha = -1;
		    const float beta = 1;
		    
		    cblas_sgemm(cblasRowmajor,TransA,TransB,mm,nn,kk,alpha,temp2,lda,temp1,ldb,beta,temp3,ldc);
		    
		    // store temp3
		    Incopy(1,c,temp3,(m-(i+1)*nb),(n-(i+1)*nb),n*nb*(i+1)+i*nb+nb,n);
		
		    mkl_free(temp1);
		    mkl_free(temp2);
	    }
	
	    // directly use LU factorization to the final matrix
	    i = i-1;
	    int mm = (m-(i+1)*nb);
	    int nn = (n-(i+1)*nb);
	    int lda = nn;
	    int b[m];
	    int info = LAPACKE_sgetrf(101,mm,nn,temp3,lda,b);
	    if(info < 0)
		{
			printf("LU factorization failed\n");
            printf("info = %d\n",info);
			mkl_free(c);
            mkl_free(temp3);
			return 0;
		}
	    else
	    {
		    Incopy(1,c,temp3,(m-(i+1)*nb),(n-(i+1)*nb),n*nb*(i+1)+i*nb+nb,n);
		    mkl_free(temp3);
	    }
    }
    
	// total time cost(seconds)
	double time = dsecnd() - start;
	CorrectOrNot(c,n);
    printf("Sizes:       %d\n",n);
    printf("nb:          %d\n",nb);
    printf("Times:       %.05f s\n",time);
	char choice;

    printf(" Show the result? Y/N \n");
	scanf("%c",&choice);
	if(choice == 'Y'|| choice == 'y')
	{
		i1 = 0;
		for(i =0 ;i<m; i++)
		{
		    for(j=0;j<n;j++)
                printf("%f\n",c[i1++]);
             printf("\n");
		}
	}
	mkl_free(c);
	return 0;
}

void Init(int N,float *A)
{
	int i,j,k;
	float ii;
	for(ii =0,i = 0;i < N;i++,ii++)
		for(j = 0;j < 2;j++)
			for(k = 0;k < ( N - i ); k++)
			{
				if(j == 0)
					A[ i * N + i + k ] = ii + 1;
				else
					A[ i * N + i +N * k ] = ii + 1;
			}
}



void Incopy(int sign,float *change,float *unchange,int out,int in,int index,int n)
{
	int i1,j,k;
	if(sign == 1)
	{
		for(i1=0,j=0;j<out;j++)
			for(k=0;k<in;k++)
				change[index+k+j*n] = unchange[i1++];
	}
	else
	{
		for(i1=0,j=0;j<out;j++)
			for(k=0;k<in;k++)
				change[i1++] = unchange[index+k+j*n];
	}
}

void CorrectOrNot(float *Array,int NN)
{
    int i;
	for( i = 0 ;i < NN*NN; i ++)
	{
		if( Array[ i ] != 1 )
		{
           printf("the result of LU is Wrong !\n");
           break;
		}
		else if(i == NN*NN-1)
            printf("the result of LU is Right!\n");
	}
}
