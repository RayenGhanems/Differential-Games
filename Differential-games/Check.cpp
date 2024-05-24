#include <iostream>
#include </opt/intel/oneapi/mkl/2024.1/include/mkl.h>

int main() {
    const int N = 1000;
    double *A = (double *)mkl_malloc(N * N * sizeof(double), 64);
    double *B = (double *)mkl_malloc(N * N * sizeof(double), 64);
    double *C = (double *)mkl_malloc(N * N * sizeof(double), 64);

    // Initialize matrices A and B

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, 1.0, A, N, B, N, 0.0, C, N);

    mkl_free(A);
    mkl_free(B);
    mkl_free(C);

    std::cout << "Matrix multiplication using MKL succeeded." << std::endl;

    return 0;
}
