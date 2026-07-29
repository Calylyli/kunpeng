#include <complex.h>
#include <stddef.h>
#include <stdlib.h>

typedef int BLASINT;
typedef double _Complex zdouble;
enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 };

#ifndef ZGEMM_MB
#define ZGEMM_MB 64
#endif
#ifndef ZGEMM_NB
#define ZGEMM_NB 32
#endif

static void zgemm_nn_row(BLASINT M, BLASINT N, BLASINT K, zdouble alpha,
                         const zdouble* restrict A, BLASINT lda,
                         const zdouble* restrict B, BLASINT ldb, zdouble beta,
                         zdouble* restrict C, BLASINT ldc)
{
    size_t packed_size = (size_t)K * N;
    double* packed = (double*)malloc(2 * packed_size * sizeof(double));
    if (!packed) return;
    double* restrict br = packed;
    double* restrict bi = packed + packed_size;
#pragma omp parallel for schedule(static)
    for (BLASINT k = 0; k < K; ++k)
        for (BLASINT j = 0; j < N; ++j) {
            zdouble value = B[(size_t)k * ldb + j];
            br[(size_t)k * N + j] = creal(value);
            bi[(size_t)k * N + j] = cimag(value);
        }

#pragma omp parallel for collapse(2) schedule(static)
    for (BLASINT ib = 0; ib < M; ib += ZGEMM_MB) {
        for (BLASINT jb = 0; jb < N; jb += ZGEMM_NB) {
            double sr[ZGEMM_MB * ZGEMM_NB], si[ZGEMM_MB * ZGEMM_NB];
            BLASINT imax = M - ib < ZGEMM_MB ? M - ib : ZGEMM_MB;
            BLASINT jmax = N - jb < ZGEMM_NB ? N - jb : ZGEMM_NB;
            for (BLASINT ii = 0; ii < imax; ++ii) {
#pragma omp simd
                for (BLASINT jj = 0; jj < jmax; ++jj) sr[ii*ZGEMM_NB+jj] = si[ii*ZGEMM_NB+jj] = 0.0;
            }
            for (BLASINT k = 0; k < K; ++k) {
                const double* restrict brr = br + (size_t)k * N + jb;
                const double* restrict bii = bi + (size_t)k * N + jb;
                for (BLASINT ii = 0; ii < imax; ++ii) {
                    zdouble av = A[(size_t)(ib + ii) * lda + k];
                    double ar = creal(av), ai = cimag(av);
                    double* restrict srr = sr + ii * ZGEMM_NB;
                    double* restrict sii = si + ii * ZGEMM_NB;
#pragma omp simd
                    for (BLASINT jj = 0; jj < jmax; ++jj) {
                        srr[jj] += ar * brr[jj] - ai * bii[jj];
                        sii[jj] += ar * bii[jj] + ai * brr[jj];
                    }
                }
            }
            for (BLASINT ii = 0; ii < imax; ++ii) {
                zdouble* restrict crow = C + (size_t)(ib + ii) * ldc + jb;
                double* restrict srr = sr + ii * ZGEMM_NB;
                double* restrict sii = si + ii * ZGEMM_NB;
                for (BLASINT jj = 0; jj < jmax; ++jj)
                    crow[jj] = alpha * (srr[jj] + sii[jj] * I) + beta * crow[jj];
            }
        }
    }
    free(packed);
}

static inline zdouble elem(const zdouble* x, BLASINT r, BLASINT c, BLASINT ld,
                           int rowmajor, enum CBLAS_TRANSPOSE t)
{
    size_t q = rowmajor ? (t == CblasNoTrans ? (size_t)r*ld+c : (size_t)c*ld+r)
                        : (t == CblasNoTrans ? (size_t)c*ld+r : (size_t)r*ld+c);
    return t == CblasConjTrans ? conj(x[q]) : x[q];
}

void cblas_zgemm(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE ta,
                 const enum CBLAS_TRANSPOSE tb, BLASINT M, BLASINT N, BLASINT K,
                 const void* alpha, const void* A, BLASINT lda, const void* B,
                 BLASINT ldb, const void* beta, void* C, BLASINT ldc)
{
    zdouble av = *(const zdouble*)alpha, bv = *(const zdouble*)beta;
    if (order == CblasRowMajor && ta == CblasNoTrans && tb == CblasNoTrans) {
        zgemm_nn_row(M,N,K,av,A,lda,B,ldb,bv,C,ldc); return;
    }
    int rm = order == CblasRowMajor;
#pragma omp parallel for schedule(static)
    for (BLASINT i=0;i<M;++i) for (BLASINT j=0;j<N;++j) {
        zdouble sum=0.0;
        for (BLASINT k=0;k<K;++k) sum += elem(A,i,k,lda,rm,ta)*elem(B,k,j,ldb,rm,tb);
        size_t q=rm?(size_t)i*ldc+j:(size_t)j*ldc+i;
        ((zdouble*)C)[q]=av*sum+bv*((zdouble*)C)[q];
    }
}
