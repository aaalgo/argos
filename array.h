#ifndef ARGOS_ARRAY
#define ARGOS_ARRAY

#include <iostream>
#include <cstdarg>
//#include <stdexcept>
#include <array>
#include <vector>
#include <algorithm>

namespace argos {
    
    using namespace std;

    using std::length_error;
    using std::array;
    using std::vector;
    using std::copy;
    using std::fill;
    using std::pair;
    using std::make_pair;

    // Multiple dimensional array.
    template <typename T = double>    // align to cache line?
    class Array {
    public:
        static constexpr size_t max_dim = 32;
        typedef T value_type;
    private:
        size_t m_dim;
        array<size_t, max_dim> m_size;   // e.g.     5, 4, 6
        array<size_t, max_dim> m_stride; // stride of one element along this dimension in storage
                                         // e.g.     32, 8, 1
        vector<T> m_data;                // m_data.size() = 256

        // initialize member data and allocate array data.
        void init (size_t dim, size_t const *size) {
            m_dim = dim;
            size_t len = 1;
            for (int i = dim-1; i >= 0; --i) {
                m_size[i] = size[i];
                m_stride[i] = len;
                len *= size[i];
            }
            m_data.resize(len);
        }
    public:
        Array (): m_dim(0) {
        }

        void display ()
        {
            cout << m_dim << ' ' << m_stride[0] << ' ' << m_stride[1] << ' ' << m_stride[2] << ' ' << m_stride[3] << endl;

        }

        void clear () {
            m_dim = 0;
            m_data.clear();
        }

        void resize (size_t dim, size_t const *size) {
            init(dim, size);
        }

        void resize (vector<size_t> const &size) {
            init(size.size(), &size[0]);
        }

        template <typename size_type>
        void resize (size_t dim, size_type d1, ...) {
            vector<size_t> size(dim);
            va_list vl;
            va_start(vl, d1);
            size[0] = d1;
            for (size_t i = 1; i < dim; ++i) {
                size[i] = va_arg(vl, size_type);
            }
            va_end(vl);
            init(dim, &size[0]);
        }


        value_type *at (size_t d1) {
            return &m_data[d1 * m_stride[0]];
        }

        value_type const *at (size_t d1) const {
            return &m_data[d1 * m_stride[0]];
        }

        value_type *at (size_t d1, size_t d2) {
            return &m_data[d1 * m_stride[0] + d2 * m_stride[1]];
        }

        value_type const *at (size_t d1, size_t d2) const {
            return &m_data[d1 * m_stride[0] + d2 * m_stride[1]];
        }
        // access element by coordinate
        /*
        template <typename size_type>
        value_type *at (size_type d1, ...) {
            va_list vl;
            va_start(vl, d1);
            size_t off = 0;
            for (size_t i = 1; i < m_dim; ++i) {
                off *= m_size[i];
                off += va_arg(vl, size_type);
            }
            va_end(vl);
            return &m_data[off];
        }

        // access element by coordinate
        template <typename size_type>
        value_type const *at (size_type d1, ...) const {
            va_list vl;
            va_start(vl, d1);
            size_t off = 0;
            for (size_t i = 1; i < m_dim; ++i) {
                off *= m_size[i];
                off += va_arg(vl, size_type);
            }
            va_end(vl);
            return &m_data[off];
        }
        */

        // walk o elements along dimension d
        template <unsigned d>
        value_type *walk (value_type *p, size_t o) {
            return p + o * m_stride[d];
        }

        template <unsigned d>
        value_type const *walk (value_type const *p, size_t o) const {
            return p + o * m_stride[d];
        }

        template <unsigned d>
        value_type *walk (value_type *p) {
            return p + m_stride[d];
        }

        template <unsigned d>
        value_type const *walk (value_type const *p) const {
            return p + m_stride[d];
        }

        value_type const *addr () const {
            return &m_data[0];
        }

        value_type *addr () {
            return &m_data[0];
        }

        size_t dim () const {
            return m_dim;
        }

        double l2 () const {
            double s = 0;
            for (auto const &v: m_data) {
                s += v * v;
            }
            return std::sqrt(s);
        }

        void size (vector<size_t> *sz) const {
            sz->resize(m_dim);
            copy(m_size.begin(), m_size.begin() + m_dim, sz->begin());
        }

        size_t size (size_t d) const {
            return m_size[d];
        }

        size_t size () const {
            return m_data.size();
        }

        void sync (Array<T> const &from) {
            BOOST_VERIFY(from.size() == size());
            copy(from.m_data.begin(), from.m_data.end(), m_data.begin());
        }

        pair<T *, T *> range () {
            return make_pair(&m_data[0], &m_data[0] + m_data.size());
        }

        // arithmetics
        void fill (T const &v) {
            std::fill(m_data.begin(), m_data.end(), v);
        }

        void scale (T const &v) {
            for (T &a: m_data) {
                a *= v;
            }
        }

        void add_diff (Array<T> const &a, Array<T> const &b) {
            BOOST_VERIFY(a.size() == size());
            BOOST_VERIFY(b.size() == size());
            for (size_t i = 0; i < m_data.size(); ++i) {
                m_data[i] += a.m_data[i] - b.m_data[i];
            }
        }

        void add_scaled (T const &a, Array<T> const &b) {
            BOOST_VERIFY(size() == b.size());
            for (size_t i = 0; i < m_data.size(); ++i) {
                m_data[i] += a * b.m_data[i];
            }
        }

        void add_scaled_wrapping (T const &scale, Array<T> const &a) {
            BOOST_VERIFY(a.m_data.size() % m_data.size() == 0);
            for (size_t i = 0; i < a.m_data.size(); i += m_data.size()) {
                /*
                for (size_t j = 0; j < 
                std::copy(a.m_data.begin(), a.m_data.end(), &m_data[i]);
                */
                for (size_t j = 0; j < m_data.size(); ++j) {
                    m_data[j] += scale * a.m_data[i + j];
                }
            }
        }

        T l2sqr (Array<T> const &a) const {
            T r = 0;
            for (size_t i = 0; i < m_data.size(); ++i) {
                T v = m_data[i] - a.m_data[i];
                r += v * v;
            }
            return r;
        }

        void tile (Array<T> const &a) {
            BOOST_VERIFY(m_data.size() % a.m_data.size() == 0);
            for (size_t i = 0; i < m_data.size(); i += a.m_data.size()) {
                std::copy(a.m_data.begin(), a.m_data.end(), &m_data[i]);
            }
        }

        template <typename OP>
        void apply (OP const &op) {
            T *y = addr();
            size_t sz = size();
#pragma omp parallel for
            for (size_t i = 0; i < sz; ++i) {
                op(y[i]);
            }
        }

        template <typename OP>
        void apply_serial (OP const &op) {
            T *y = addr();
            size_t sz = size();
            for (size_t i = 0; i < sz; ++i) {
                op(*y);
                ++y;;
            }
        }

        template <typename OP>
        void apply (Array<T> const &ax, OP const &op) {
            T const *x = ax.addr();
            T *y = addr();
            size_t sz = size();
#pragma omp parallel for
            for (size_t i = 0; i < sz; ++i) {
                op(y[i], x[i]);
            }
        }

        template <typename OP>
        void apply (Array<T> const &ax1, Array<T> const &ax2, OP const &op) {
            T const *x1 = ax1.addr();
            T const *x2 = ax2.addr();
            T *y = addr();
            size_t sz = size();
#pragma omp parallel for
            for (size_t i = 0; i < sz; ++i) {
                op(y[i], x1[i], x2[i]);
            }
        }

        template <typename OP>
        void apply (Array<T> const &ax1, Array<T> const &ax2, Array<T> const &ax3, OP const &op) {
            T const *x1 = ax1.addr();
            T const *x2 = ax2.addr();
            T const *x3 = ax3.addr();
            T *y = addr();
            size_t sz = size();
#pragma omp parallel for
            for (size_t i = 0; i < sz; ++i) {
                op(y[i], x1[i], x2[i], x3[i]);
            }
        }
    };
}

#endif
