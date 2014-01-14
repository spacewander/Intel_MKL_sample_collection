// for dot product between two vectors
#include <stdio.h> 
#include <mkl.h>
#define N 5

int main()
{
    int n;
    int inca = 1;
    int incb = 2;
    int i;
    MKL_Complex16 a[N], b[N],c;
    n = N;
    for (i = 0; i < n; ++i)
    {
        a[i].real = ( double )i * 2.0;
        a[i].imag = ( double )i * 2.0;
        b[i].real = 2 * ( double )(n - i);
        b[i].imag = ( double )i * 2.0;
    }
    zdotc( &c, &n, a, &inca, b, &incb);
    printf("The complex dot product is:( %6.2f, %6.2f )\n",c.real,c.imag);
    return 0;
}
