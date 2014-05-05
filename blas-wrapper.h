#ifndef ARGOS_BLAS
#define ARGOS_BLAS

#include <cblas.h>

namespace argos {
    namespace blas {
        template <typename T>
        void gemm (T const *A, size_t A_rows, size_t A_cols, bool transA,
                   T const *B, size_t B_rows, size_t B_cols, bool transB,
                   T *C, size_t C_rows, size_t C_cols, T alpha, T beta);
    }
}

#endif
