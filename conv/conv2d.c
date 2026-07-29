#include <string.h>
#include <omp.h>

// ==================== 类型定义 ====================
typedef float CONVFLOAT;
typedef int CONVINT;

#ifndef CONV_COLUMN_BLOCK
#define CONV_COLUMN_BLOCK 8192
#endif

/*
 * Block the output columns so that the partial sums stay in L1 while a whole
 * kernel is applied.  Iterating across columns in the innermost loop exposes
 * contiguous input and output vectors to NEON.  The jk/ik order is unchanged,
 * which also keeps the floating-point accumulation order of each output.
 */
void conv2d(const CONVFLOAT* input, CONVINT inputHeight, CONVINT inputWidth,
            const CONVFLOAT* kernel, CONVINT kernelHeight, CONVINT kernelWidth, CONVFLOAT* output)
{
    const CONVINT outputHeight = inputHeight - kernelHeight + 1;
    const CONVINT outputWidth = inputWidth - kernelWidth + 1;
    const CONVINT columnBlock = CONV_COLUMN_BLOCK;

#pragma omp parallel for schedule(static)
    for (CONVINT j = 0; j < outputHeight; ++j) {
        CONVFLOAT* restrict outputRow = output + (size_t)j * outputWidth;

#pragma omp simd
        for (CONVINT i = 0; i < outputWidth; ++i) {
            outputRow[i] = 0.0f;
        }

        for (CONVINT ib = 0; ib < outputWidth; ib += columnBlock) {
            const CONVINT end = ib + columnBlock < outputWidth ? ib + columnBlock : outputWidth;
            for (CONVINT jk = 0; jk < kernelHeight; ++jk) {
                const CONVFLOAT* restrict inputRow = input + (size_t)(j + jk) * inputWidth;
                const CONVFLOAT* restrict kernelRow = kernel + (size_t)jk * kernelWidth;
                for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                    const CONVFLOAT coefficient = kernelRow[ik];
#pragma omp simd
                    for (CONVINT i = ib; i < end; ++i) {
                        outputRow[i] += inputRow[i + ik] * coefficient;
                    }
                }
            }
        }
    }
}
