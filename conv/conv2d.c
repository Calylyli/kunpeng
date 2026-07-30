#include <string.h>
#include <omp.h>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

// ==================== 类型定义 ====================
typedef float CONVFLOAT;
typedef int CONVINT;

/*
 * Keep a short output strip in NEON registers while applying the entire
 * kernel.  Every output still accumulates in jk/ik order, matching the
 * reference implementation, but is written to memory only once.
 */
void conv2d(const CONVFLOAT* input, CONVINT inputHeight, CONVINT inputWidth,
            const CONVFLOAT* kernel, CONVINT kernelHeight, CONVINT kernelWidth, CONVFLOAT* output)
{
    const CONVINT outputHeight = inputHeight - kernelHeight + 1;
    const CONVINT outputWidth = inputWidth - kernelWidth + 1;

#pragma omp parallel for schedule(static)
    for (CONVINT j = 0; j < outputHeight; ++j) {
        CONVFLOAT* restrict outputRow = output + (size_t)j * outputWidth;
#if defined(__aarch64__)
        CONVINT ib = 0;
        for (; ib + 95 < outputWidth; ib += 96) {
            float32x4_t s0 = vdupq_n_f32(0.0f);
            float32x4_t s1 = vdupq_n_f32(0.0f);
            float32x4_t s2 = vdupq_n_f32(0.0f);
            float32x4_t s3 = vdupq_n_f32(0.0f);
            float32x4_t s4 = vdupq_n_f32(0.0f);
            float32x4_t s5 = vdupq_n_f32(0.0f);
            float32x4_t s6 = vdupq_n_f32(0.0f);
            float32x4_t s7 = vdupq_n_f32(0.0f);
            float32x4_t s8 = vdupq_n_f32(0.0f);
            float32x4_t s9 = vdupq_n_f32(0.0f);
            float32x4_t s10 = vdupq_n_f32(0.0f);
            float32x4_t s11 = vdupq_n_f32(0.0f);
            float32x4_t s12 = vdupq_n_f32(0.0f);
            float32x4_t s13 = vdupq_n_f32(0.0f);
            float32x4_t s14 = vdupq_n_f32(0.0f);
            float32x4_t s15 = vdupq_n_f32(0.0f);
            float32x4_t s16 = vdupq_n_f32(0.0f);
            float32x4_t s17 = vdupq_n_f32(0.0f);
            float32x4_t s18 = vdupq_n_f32(0.0f);
            float32x4_t s19 = vdupq_n_f32(0.0f);
            float32x4_t s20 = vdupq_n_f32(0.0f);
            float32x4_t s21 = vdupq_n_f32(0.0f);
            float32x4_t s22 = vdupq_n_f32(0.0f);
            float32x4_t s23 = vdupq_n_f32(0.0f);
            for (CONVINT jk = 0; jk < kernelHeight; ++jk) {
                const CONVFLOAT* restrict inputRow = input + (size_t)(j + jk) * inputWidth;
                const CONVFLOAT* restrict kernelRow = kernel + (size_t)jk * kernelWidth;
                for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                    const CONVFLOAT coefficient = kernelRow[ik];
                    const CONVFLOAT* restrict in = inputRow + ib + ik;
                    s0 = vfmaq_n_f32(s0, vld1q_f32(in), coefficient);
                    s1 = vfmaq_n_f32(s1, vld1q_f32(in + 4), coefficient);
                    s2 = vfmaq_n_f32(s2, vld1q_f32(in + 8), coefficient);
                    s3 = vfmaq_n_f32(s3, vld1q_f32(in + 12), coefficient);
                    s4 = vfmaq_n_f32(s4, vld1q_f32(in + 16), coefficient);
                    s5 = vfmaq_n_f32(s5, vld1q_f32(in + 20), coefficient);
                    s6 = vfmaq_n_f32(s6, vld1q_f32(in + 24), coefficient);
                    s7 = vfmaq_n_f32(s7, vld1q_f32(in + 28), coefficient);
                    s8 = vfmaq_n_f32(s8, vld1q_f32(in + 32), coefficient);
                    s9 = vfmaq_n_f32(s9, vld1q_f32(in + 36), coefficient);
                    s10 = vfmaq_n_f32(s10, vld1q_f32(in + 40), coefficient);
                    s11 = vfmaq_n_f32(s11, vld1q_f32(in + 44), coefficient);
                    s12 = vfmaq_n_f32(s12, vld1q_f32(in + 48), coefficient);
                    s13 = vfmaq_n_f32(s13, vld1q_f32(in + 52), coefficient);
                    s14 = vfmaq_n_f32(s14, vld1q_f32(in + 56), coefficient);
                    s15 = vfmaq_n_f32(s15, vld1q_f32(in + 60), coefficient);
                    s16 = vfmaq_n_f32(s16, vld1q_f32(in + 64), coefficient);
                    s17 = vfmaq_n_f32(s17, vld1q_f32(in + 68), coefficient);
                    s18 = vfmaq_n_f32(s18, vld1q_f32(in + 72), coefficient);
                    s19 = vfmaq_n_f32(s19, vld1q_f32(in + 76), coefficient);
                    s20 = vfmaq_n_f32(s20, vld1q_f32(in + 80), coefficient);
                    s21 = vfmaq_n_f32(s21, vld1q_f32(in + 84), coefficient);
                    s22 = vfmaq_n_f32(s22, vld1q_f32(in + 88), coefficient);
                    s23 = vfmaq_n_f32(s23, vld1q_f32(in + 92), coefficient);
                }
            }
            vst1q_f32(outputRow + ib, s0);
            vst1q_f32(outputRow + ib + 4, s1);
            vst1q_f32(outputRow + ib + 8, s2);
            vst1q_f32(outputRow + ib + 12, s3);
            vst1q_f32(outputRow + ib + 16, s4);
            vst1q_f32(outputRow + ib + 20, s5);
            vst1q_f32(outputRow + ib + 24, s6);
            vst1q_f32(outputRow + ib + 28, s7);
            vst1q_f32(outputRow + ib + 32, s8);
            vst1q_f32(outputRow + ib + 36, s9);
            vst1q_f32(outputRow + ib + 40, s10);
            vst1q_f32(outputRow + ib + 44, s11);
            vst1q_f32(outputRow + ib + 48, s12);
            vst1q_f32(outputRow + ib + 52, s13);
            vst1q_f32(outputRow + ib + 56, s14);
            vst1q_f32(outputRow + ib + 60, s15);
            vst1q_f32(outputRow + ib + 64, s16);
            vst1q_f32(outputRow + ib + 68, s17);
            vst1q_f32(outputRow + ib + 72, s18);
            vst1q_f32(outputRow + ib + 76, s19);
            vst1q_f32(outputRow + ib + 80, s20);
            vst1q_f32(outputRow + ib + 84, s21);
            vst1q_f32(outputRow + ib + 88, s22);
            vst1q_f32(outputRow + ib + 92, s23);
        }

#pragma omp simd
        for (CONVINT i = ib; i < outputWidth; ++i) outputRow[i] = 0.0f;
        for (CONVINT jk = 0; jk < kernelHeight; ++jk) {
            const CONVFLOAT* restrict inputRow = input + (size_t)(j + jk) * inputWidth;
            const CONVFLOAT* restrict kernelRow = kernel + (size_t)jk * kernelWidth;
            for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                const CONVFLOAT coefficient = kernelRow[ik];
#pragma omp simd
                for (CONVINT i = ib; i < outputWidth; ++i)
                    outputRow[i] += inputRow[i + ik] * coefficient;
            }
        }
#else
#pragma omp simd
        for (CONVINT i = 0; i < outputWidth; ++i) outputRow[i] = 0.0f;
        for (CONVINT jk = 0; jk < kernelHeight; ++jk) {
            const CONVFLOAT* restrict inputRow = input + (size_t)(j + jk) * inputWidth;
            const CONVFLOAT* restrict kernelRow = kernel + (size_t)jk * kernelWidth;
            for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                const CONVFLOAT coefficient = kernelRow[ik];
#pragma omp simd
                for (CONVINT i = 0; i < outputWidth; ++i) {
                    outputRow[i] += inputRow[i + ik] * coefficient;
                }
            }
        }
#endif
    }
}
