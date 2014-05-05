#include <boost/assert.hpp>
#include "blas-wrapper.h"

namespace argos {
    namespace blas {
        template <>
        void gemm (float const *A, size_t A_rows, size_t A_cols, bool transA,
                   float const *B, size_t B_rows, size_t B_cols, bool transB,
                   float *C, size_t C_rows, size_t C_cols, float alpha, float beta) {
            size_t M, N, K;
            enum CBLAS_TRANSPOSE TA, TB;
            if (transA) {
                M = A_cols;
                K = A_rows;
                TA = CblasTrans;
            } else {
                M = A_rows;
                K = A_cols;
                TA = CblasNoTrans;
            }

            if (transB) {
                N = B_rows;
                BOOST_VERIFY(K == B_cols);
                TB = CblasTrans;
            } else {
                N = B_cols;
                BOOST_VERIFY(K == B_rows);
                TB = CblasNoTrans;
            }
            BOOST_VERIFY(M == C_rows);
            BOOST_VERIFY(N == C_cols);
            cblas_sgemm(CblasRowMajor, TA, TB, M, N, K, alpha, A, A_cols, B, B_cols, beta, C, C_cols);
        }

        template <>
        void gemm (double const *A, size_t A_rows, size_t A_cols, bool transA,
                   double const *B, size_t B_rows, size_t B_cols, bool transB,
                   double *C, size_t C_rows, size_t C_cols, double alpha, double beta) {
            size_t M, N, K;
            enum CBLAS_TRANSPOSE TA, TB;
            if (transA) {
                M = A_cols;
                K = A_rows;
                TA = CblasTrans;
            } else {
                M = A_rows;
                K = A_cols;
                TA = CblasNoTrans;
            }

            if (transB) {
                N = B_rows;
                BOOST_VERIFY(K == B_cols);
                TB = CblasTrans;
            } else {
                N = B_cols;
                BOOST_VERIFY(K == B_rows);
                TB = CblasNoTrans;
            }
            BOOST_VERIFY(M == C_rows);
            BOOST_VERIFY(N == C_cols);
            cblas_dgemm(CblasRowMajor, TA, TB, M, N, K, alpha, A, A_cols, B, B_cols, beta, C, C_cols);
        }
    }
}
