#include <complex.h>
#include <stddef.h>
#include <stdlib.h>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

typedef int BLASINT;
typedef double _Complex zdouble;
enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 };

#ifndef ZGEMM_MB
#define ZGEMM_MB 60
#endif
#ifndef ZGEMM_NB
#define ZGEMM_NB 16
#endif
#ifndef ZGEMM_BLOCKED_MB
#define ZGEMM_BLOCKED_MB 64
#endif
#ifndef ZGEMM_BLOCKED_NB
#define ZGEMM_BLOCKED_NB 32
#endif

static void zgemm_blocked(BLASINT M, BLASINT N, BLASINT K, zdouble alpha,
                          const zdouble* restrict A, BLASINT lda,
                          const double* restrict br, const double* restrict bi,
                          zdouble beta, zdouble* restrict C, BLASINT ldc)
{
#pragma omp parallel for collapse(2) schedule(static)
    for (BLASINT ib = 0; ib < M; ib += ZGEMM_BLOCKED_MB) {
        for (BLASINT jb = 0; jb < N; jb += ZGEMM_BLOCKED_NB) {
            double sr[ZGEMM_BLOCKED_MB * ZGEMM_BLOCKED_NB];
            double si[ZGEMM_BLOCKED_MB * ZGEMM_BLOCKED_NB];
            BLASINT imax = M - ib < ZGEMM_BLOCKED_MB ? M - ib : ZGEMM_BLOCKED_MB;
            BLASINT jmax = N - jb < ZGEMM_BLOCKED_NB ? N - jb : ZGEMM_BLOCKED_NB;
            for (BLASINT ii = 0; ii < imax; ++ii) {
#pragma omp simd
                for (BLASINT jj = 0; jj < jmax; ++jj)
                    sr[ii * ZGEMM_BLOCKED_NB + jj] =
                        si[ii * ZGEMM_BLOCKED_NB + jj] = 0.0;
            }
            for (BLASINT k = 0; k < K; ++k) {
                const double* restrict brr = br + (size_t)k * N + jb;
                const double* restrict bii = bi + (size_t)k * N + jb;
                for (BLASINT ii = 0; ii < imax; ++ii) {
                    zdouble av = A[(size_t)(ib + ii) * lda + k];
                    double ar = creal(av), ai = cimag(av);
                    double* restrict srr = sr + ii * ZGEMM_BLOCKED_NB;
                    double* restrict sii = si + ii * ZGEMM_BLOCKED_NB;
#pragma omp simd
                    for (BLASINT jj = 0; jj < jmax; ++jj) {
                        srr[jj] += ar * brr[jj] - ai * bii[jj];
                        sii[jj] += ar * bii[jj] + ai * brr[jj];
                    }
                }
            }
            for (BLASINT ii = 0; ii < imax; ++ii) {
                zdouble* restrict crow = C + (size_t)(ib + ii) * ldc + jb;
                double* restrict srr = sr + ii * ZGEMM_BLOCKED_NB;
                double* restrict sii = si + ii * ZGEMM_BLOCKED_NB;
                for (BLASINT jj = 0; jj < jmax; ++jj)
                    crow[jj] = alpha * (srr[jj] + sii[jj] * I) + beta * crow[jj];
            }
        }
    }
}

#if defined(__aarch64__)
static inline void zgemm_kernel_6x4(BLASINT K, const zdouble* restrict A,
                                    BLASINT lda, const double* restrict br,
                                    const double* restrict bi, BLASINT N,
                                    zdouble alpha, zdouble beta,
                                    zdouble* restrict C, BLASINT ldc)
{
    float64x2_t r00 = vdupq_n_f64(0.0), r01 = vdupq_n_f64(0.0);
    float64x2_t i00 = vdupq_n_f64(0.0), i01 = vdupq_n_f64(0.0);
    float64x2_t r10 = vdupq_n_f64(0.0), r11 = vdupq_n_f64(0.0);
    float64x2_t i10 = vdupq_n_f64(0.0), i11 = vdupq_n_f64(0.0);
    float64x2_t r20 = vdupq_n_f64(0.0), r21 = vdupq_n_f64(0.0);
    float64x2_t i20 = vdupq_n_f64(0.0), i21 = vdupq_n_f64(0.0);
    float64x2_t r30 = vdupq_n_f64(0.0), r31 = vdupq_n_f64(0.0);
    float64x2_t i30 = vdupq_n_f64(0.0), i31 = vdupq_n_f64(0.0);
    float64x2_t r40 = vdupq_n_f64(0.0), r41 = vdupq_n_f64(0.0);
    float64x2_t i40 = vdupq_n_f64(0.0), i41 = vdupq_n_f64(0.0);
    float64x2_t r50 = vdupq_n_f64(0.0), r51 = vdupq_n_f64(0.0);
    float64x2_t i50 = vdupq_n_f64(0.0), i51 = vdupq_n_f64(0.0);

    for (BLASINT k = 0; k < K; ++k) {
        const float64x2_t brr0 = vld1q_f64(br + (size_t)k * N);
        const float64x2_t brr1 = vld1q_f64(br + (size_t)k * N + 2);
        const float64x2_t bii0 = vld1q_f64(bi + (size_t)k * N);
        const float64x2_t bii1 = vld1q_f64(bi + (size_t)k * N + 2);

#define ZGEMM_ROW(ROW, R0, R1, I0, I1) do { \
        const zdouble a = A[(size_t)(ROW) * lda + k]; \
        const double ar = creal(a), ai = cimag(a); \
        R0 = vfmaq_n_f64(R0, brr0, ar); \
        R0 = vfmsq_n_f64(R0, bii0, ai); \
        R1 = vfmaq_n_f64(R1, brr1, ar); \
        R1 = vfmsq_n_f64(R1, bii1, ai); \
        I0 = vfmaq_n_f64(I0, bii0, ar); \
        I0 = vfmaq_n_f64(I0, brr0, ai); \
        I1 = vfmaq_n_f64(I1, bii1, ar); \
        I1 = vfmaq_n_f64(I1, brr1, ai); \
    } while (0)
        ZGEMM_ROW(0, r00, r01, i00, i01);
        ZGEMM_ROW(1, r10, r11, i10, i11);
        ZGEMM_ROW(2, r20, r21, i20, i21);
        ZGEMM_ROW(3, r30, r31, i30, i31);
        ZGEMM_ROW(4, r40, r41, i40, i41);
        ZGEMM_ROW(5, r50, r51, i50, i51);
#undef ZGEMM_ROW
    }

#define ZGEMM_STORE(ROW, R0, R1, I0, I1) do { \
        double rr[4], ii[4]; \
        vst1q_f64(rr, R0); vst1q_f64(rr + 2, R1); \
        vst1q_f64(ii, I0); vst1q_f64(ii + 2, I1); \
        zdouble* restrict cp = C + (size_t)(ROW) * ldc; \
        for (BLASINT q = 0; q < 4; ++q) \
            cp[q] = alpha * (rr[q] + ii[q] * I) + beta * cp[q]; \
    } while (0)
    ZGEMM_STORE(0, r00, r01, i00, i01);
    ZGEMM_STORE(1, r10, r11, i10, i11);
    ZGEMM_STORE(2, r20, r21, i20, i21);
    ZGEMM_STORE(3, r30, r31, i30, i31);
    ZGEMM_STORE(4, r40, r41, i40, i41);
    ZGEMM_STORE(5, r50, r51, i50, i51);
#undef ZGEMM_STORE
}
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

#if defined(__aarch64__)
    /* The register kernel wins while the packed panels fit comfortably in
       cache.  Large output matrices retain the wider cache-blocked kernel. */
    if ((size_t)M * N > 100000000u) {
        zgemm_blocked(M, N, K, alpha, A, lda, br, bi, beta, C, ldc);
        free(packed);
        return;
    }
#pragma omp parallel for collapse(2) schedule(static)
    for (BLASINT ib = 0; ib < M; ib += ZGEMM_MB) {
        for (BLASINT jb = 0; jb < N; jb += ZGEMM_NB) {
            const BLASINT m6 = M - M % 6;
            const BLASINT iend = ib + ZGEMM_MB < m6 ? ib + ZGEMM_MB : m6;
            const BLASINT jend = jb + ZGEMM_NB < (N & ~3)
                                     ? jb + ZGEMM_NB : (N & ~3);
            for (BLASINT jj = jb; jj < jend; jj += 4) {
                for (BLASINT ii = ib; ii + 5 < iend; ii += 6)
                    zgemm_kernel_6x4(K, A + (size_t)ii * lda, lda,
                                     br + jj, bi + jj, N, alpha, beta,
                                     C + (size_t)ii * ldc + jj, ldc);
            }
        }
    }

#pragma omp parallel for schedule(static)
    for (BLASINT i = 0; i < M; ++i) {
        const BLASINT jbegin = i < (M - M % 6) ? (N & ~3) : 0;
        for (BLASINT j = jbegin; j < N; ++j) {
            double sr = 0.0, si = 0.0;
            for (BLASINT k = 0; k < K; ++k) {
                zdouble av = A[(size_t)i * lda + k];
                double ar = creal(av), ai = cimag(av);
                double xr = br[(size_t)k * N + j];
                double xi = bi[(size_t)k * N + j];
                sr += ar * xr - ai * xi;
                si += ar * xi + ai * xr;
            }
            zdouble* cp = C + (size_t)i * ldc + j;
            *cp = alpha * (sr + si * I) + beta * *cp;
        }
    }
#else
    zgemm_blocked(M, N, K, alpha, A, lda, br, bi, beta, C, ldc);
#endif
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
