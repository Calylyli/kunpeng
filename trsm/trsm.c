/* CBLAS integer enum values are shared by KBLAS and standard CBLAS. */
enum {
    CblasRowMajor = 101,
    CblasNoTrans = 111,
    CblasLeft = 141,
    CblasLower = 122,
    CblasNonUnit = 131
};

extern void cblas_dtrsm(const int order, const int side, const int uplo,
                        const int transa, const int diag, const int m,
                        const int n, const double alpha, const double* a,
                        const int lda, double* b, const int ldb);

void l_trsm(int m, int n, const double* L, int lda, double* B, int ldb)
{
    cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans,
                CblasNonUnit, m, n, 1.0, L, lda, B, ldb);
}
