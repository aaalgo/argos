#ifndef ARGOS_NODE_CORE
#define ARGOS_NODE_CORE

#include <iostream>
#include <sstream>
#include <boost/lexical_cast.hpp>
#include "array.h"
#include "argos.h"
#include "blas-wrapper.h"

namespace argos {

    namespace role {
        class ArrayLabelInput: public virtual Role {
        public:
            virtual Array<double> const &labels () const = 0;
        };
    }


    namespace core {
        class Meta: public Node {
            double m_mom;
            double m_eta;
            double m_lambda;
        public:
            Meta (Model *model, Config const &config)
                : Node(model, config),
                  m_mom(getConfig<double>("mom", "argos.global.mom", 0)),
                  m_eta(getConfig<double>("eta", "argos.global.eta", 0.0005)),
                  m_lambda(getConfig<double>("lambda", "argos.global.lambda", 0.5))
            {
            }
            double mom () const {
                return m_mom;
            }
            double eta () const {
                return m_eta;
            }
            double lambda () const {
                return m_lambda;
            }
        };

        class ArrayNode: public Node {
        public:
            enum {
                FLAT = 0,
                SOUND = 1,
                IMAGE = 2,
            };
        private:
            vector<size_t> m_size;
            Array<> m_data;
            Array<> m_delta;
            int m_type;
        protected:
            void resize (ArrayNode const &node) {
                m_size = node.m_size;
                m_data.resize(m_size);
                m_delta.resize(m_size);
            }
            void resize (vector<size_t> const &size) {
                m_size = size;
                m_data.resize(size);
                m_delta.resize(size);
            }
            void setType (int type) {
                m_type = type;
            }
        public:
            ArrayNode (Model *model, Config const &config) : Node(model, config), m_type(FLAT) {
            }
            vector<size_t> const& size () const { return m_size; }
            Array<> &data () { return m_data; }
            Array<> &delta () { return m_delta; }
            Array<> const &data () const { return m_data; }
            Array<> const &delta () const { return m_delta; }
            void preupdate ()  {
                delta().fill(0);
            }
            int type () const {
                return m_type;
            }
            void report () const {
                cerr << name() << ":\t" << data().l2() << "\t" <<  delta().l2() << endl;
            }

            virtual void handle (http::server::request const &req, http::server::reply &rep) const {
                rep.status = http::server::reply::ok;
                ostringstream ss;
                size_t sz = data().size();
                size_t samples = data().size(size_t(0));
                size_t dim = sz / samples;
                for (unsigned i = 0; i < samples; ++i) {
                    Array<>::value_type const *x = data().at(i);
                    for (unsigned j = 0; j < dim; ++j) {
                        if (j) ss << '\t';
                        ss << x[j];
                    }
                    ss << endl;
                }
                rep.content = ss.str();
                rep.headers.resize(2);
                rep.headers[0].name = "Content-Length";
                rep.headers[0].value = boost::lexical_cast<string>(rep.content.size());
                rep.headers[1].name = "Content-Type";
                rep.headers[1].value = "text/plain";
            }
        };

        /*
        class InputNode: public ArrayNode {
        public:
            InputNode (Model *model, Config const &config): ArrayNode(model, config) {
                vector<size_t> size;
                size.push_back(model->config().get<size_t>("argos.batch"));
                try {
                    try {
                        size.push_back(config.get<size_t>("height"));
                        setType(IMAGE);
                    }
                    catch (...) {
                        setType(SOUND);
                    }
                    size.push_back(config.get<size_t>("width"));
                }
                catch (...) {
                    // default is flat
                }
                size.push_back(config.get<size_t>("channel"));
                resize(size);
            }
        };
        */

        /*
        class GaussianOutputNode: public ArrayOutputNode {
            ArrayNode *m_input;
        public:
            GaussianOutputNode (Model *model, Config const &config)
                : ArrayOutputNode(model, config) {
                m_input = findInputAndAdd<ArrayNode>("input", "input");
                resize(*m_input);
            }

            void update (Mode mode) {
                m_input->delta().add_diff(m_input->data(), this->data());
            }

            void cost (vector<double> *c) const {
                c->resize(1);
                c->at(0) = data().l2sqr(m_input->data()) / 2;
            }
        };
        */

        class MaxScoreOutputNode: public LabelOutputNode<int>
        {
        protected:
            ArrayNode *m_input;
            size_t m_samples;
            size_t m_stride;

        public:
            MaxScoreOutputNode (Model *model, Config const &config) 
                : LabelOutputNode(model, config)
             {
                m_input = findInputAndAdd<ArrayNode>("input", "input");
                m_samples = m_input->data().size(size_t(0));
                BOOST_VERIFY(m_input->data().size() % m_samples == 0);
                m_stride = m_input->data().size() / m_samples;
            }

            void predict () {
                m_labels.resize(m_samples);
                Array<>::value_type const *x = m_input->data().addr();
                for (size_t i = 0; i < m_samples; ++i) {
                    int best = 0;
                    for (unsigned l = 1; l < m_stride; ++l) {
                        if (x[l] > x[best]) {
                            best = l;
                        }
                    }
                    m_labels[i] = best;
                    x += m_stride;
                }
            }
        };

        // 
        class LogPOutputNode: public MaxScoreOutputNode, public role::Loss {
        public:
            LogPOutputNode (Model *model, Config const &config) 
                : MaxScoreOutputNode(model, config)
            {
                  role::Loss::init({"loss", "error"});
            }

            void predict () {
                MaxScoreOutputNode::predict();
                Array<>::value_type const *x = m_input->data().addr();
                vector<int> const &truth = inputLabels();
                m_labels.resize(truth.size());
                for (size_t i = 0; i < truth.size(); ++i) {
                    Array<>::value_type mm = x[0];
                    int cmax = 1;
                    for (size_t j = 1; j < m_stride; ++j) {
                        if (x[j] > mm) {
                            mm = x[j];
                            cmax = 1;
                        }
                        else if (x[j] == mm) {
                            ++cmax;
                        }
                    }
                    int l = truth[i];
                    acc(0)(-log(x[l]));
                    if (x[l] < mm) {
                        acc(1)(1.0);
                    }
                    else {
                        acc(1)(1.0 - 1.0/cmax);
                    }
                    x += m_stride;
                }
            }

            void update () {
                Array<>::value_type const *x = m_input->data().addr();
                Array<>::value_type *dx = m_input->delta().addr();
                vector<int> const &truth = inputLabels();
                for (size_t i = 0; i < m_samples; ++i) {
                    int l = truth[i];
                    dx[l] +=  -1.0/x[l];
                    x += m_stride;
                    dx += m_stride;
                }
            }
        };

        class HingeLossOutputNode: public MaxScoreOutputNode, public role::Loss {
            double m_margin;
        public:
            HingeLossOutputNode (Model *model, Config const &config) 
                : MaxScoreOutputNode(model, config),
                  m_margin(config.get<double>("margin", 0.25))
            {
                  role::Loss::init({"loss", "error"});
            }

            void predict () {
                MaxScoreOutputNode::predict();
                Array<>::value_type const *x = m_input->data().addr();
                vector<int> const &truth = inputLabels();
                m_labels.resize(truth.size());
                for (size_t i = 0; i < truth.size(); ++i) {
                    int l = truth[i];
                    unsigned bad = 0;            // sizeof left set
                    double t = x[l];
                    double total = 0;
                    for (unsigned c = 0; c < m_stride; ++c) {
                        if (c == unsigned(l)) continue;
                        if (x[c] >= t) ++bad;
                        if (((x[c] + m_margin) >= t)) {
                            total += x[c] + m_margin - t;
                        }
                    }
                    acc(0)(total);
                    acc(1)(bad ? 1.0 : 0.0);
                    x += m_stride;
                }
            }

            void update () {
                Array<>::value_type const *x = m_input->data().addr();
                Array<>::value_type *dx = m_input->delta().addr();
                vector<int> const &truth = inputLabels();
                for (size_t i = 0; i < m_samples; ++i) {
                    int l = truth[i];
                    unsigned left = 0;            // sizeof left set
                    double t = x[l];
                    for (unsigned c = 0; c < m_stride; ++c) {
                        if (c == unsigned(l)) continue;
                        if (((x[c] + m_margin) >= t)) {
                            dx[c] += 1.0;
                            ++left;
                        }
                    }
                    dx[l] += -1.0 * left;
                    x += m_stride;
                    dx += m_stride;
                }
            }
        };

        /**
         * loss = 0.5 | x - t| ^2
         * d loss/ /dx = x - t 
         */
        class RegressionOutputNode: public LabelOutputNode<double>, public role::Loss {
            ArrayNode *m_input;
            double m_margin;
        public:
            RegressionOutputNode (Model *model, Config const &config) 
                : LabelOutputNode<double>(model, config),
                  m_input(findInputAndAdd<ArrayNode>("input", "input")),
                  m_margin(config.get<double>("margin", 0))
            {
                role::Loss::init({"loss", "error"});
            }

            void predict () {
                Array<>::value_type const *x = m_input->data().addr();
                vector<double> const &truth = inputLabels();
                m_labels.resize(truth.size());
                for (size_t i = 0; i < truth.size(); ++i) {
                    m_labels[i] = x[i];
                    double diff = x[i] - truth[i];
                    double n2 = diff * diff;
                    if (std::abs(diff) >= m_margin) {
                        acc(0)(0.5 * n2);
                    }
                    else {
                        acc(0)(0);
                    }
                    acc(1)(std::abs(diff));
                }
            }

            void update () {
                Array<>::value_type const *x = m_input->data().addr();
                Array<>::value_type *dx = m_input->delta().addr();
                vector<double> const &truth = inputLabels();
                for (size_t i = 0; i < truth.size(); ++i) {
                    double diff = x[i] - truth[i];
                    if (std::abs(diff) >= m_margin) {
                        dx[i] = diff;
                    }
                    else {
                        dx[i] = 0;
                    }
                }
            }
        };

        class ArrayOutputNode: public ArrayNode { 
            role::ArrayLabelInput const *m_label_input; 
        public:
            ArrayOutputNode (Model *model, Config const &config): ArrayNode(model, config)
            {
                m_label_input = model->findNode<role::ArrayLabelInput>(config.get<string>("label"));
                BOOST_VERIFY(m_label_input);
            }
            Array<double> const &inputLabels () const { return m_label_input->labels(); }
        };

        class MultiRegressionOutputNode: public ArrayOutputNode, public role::Loss {
            ArrayNode *m_input;
            double m_margin;
            double m_rho;
        public:
            MultiRegressionOutputNode (Model *model, Config const &config) 
                : ArrayOutputNode(model, config),
                  m_input(findInputAndAdd<ArrayNode>("input", "input")),
                  m_margin(config.get<double>("margin", 0)),
                  m_rho(config.get<double>("rho", 1.0))
            {
                role::Loss::init({"loss", "error"});
            }

            void predict () {
                Array<>::value_type const *x = m_input->data().addr();
                Array<>::value_type const *y = inputLabels().addr();
                vector<size_t> sz_x;
                vector<size_t> sz_y;
                m_input->data().size(&sz_x);
                inputLabels().size(&sz_y);
                BOOST_VERIFY(sz_x.size() == 2);
                BOOST_VERIFY(sz_x == sz_y);
                unsigned i = 0;
                for (unsigned row = 0; row < sz_x[0]; ++row) {
                    double l = 0;
                    for (unsigned col = 0; col < sz_x[1]; ++col) {
                        double diff = std::abs(x[i] - y[i]);
                        ++i;
                        double n2 = diff * diff;
                        if (diff >= m_margin) {
                            l += 0.5 * n2;
                        }
                        acc(1)(diff);
                    }
                    acc(0)(l);
                }
            }

            void update () {
                Array<>::value_type const *x = m_input->data().addr();
                Array<>::value_type *dx = m_input->delta().addr();
                Array<>::value_type const *y = inputLabels().addr();
                vector<size_t> sz_x;
                vector<size_t> sz_y;
                m_input->data().size(&sz_x);
                inputLabels().size(&sz_y);
                BOOST_VERIFY(sz_x.size() == 2);
                BOOST_VERIFY(sz_x == sz_y);
                size_t total = sz_x[0] * sz_x[1];

                for (size_t i = 0; i < total; ++i) {
                    double diff = x[i] - y[i];
                    if (std::abs(diff) >= m_margin) {
                        dx[i] = diff * m_rho;
                    }
                    else {
                        dx[i] = 0;
                    }
                }
            }
        };

        namespace function {
            // each struct implement a forward function
            // which is the activate function itself,
            // and a backward function, which is the derivative
            // of the activate function as a function of y.
            //
            struct id { // identity, for testing
                static string name () {
                    return "id";
                }
                template <typename T>
                static T forward (T x) {
                    return x;
                }
                template <typename T>
                static T backward (T x, T y) {
                    return 1;
                }
            };
            struct relu {
                static string name () {
                    return "relu";
                }
                template <typename T>
                static T forward (T x) {
                    return x > 0 ? x : 0;
                }
                template <typename T>
                static T backward (T x, T y) {
                    return x > 0 ? 1 : 0;
                }
            };
            struct softrelu {
                static string name () {
                    return "softrelu";
                }
                template <typename T>
                static T forward (T x) {
                    return log(1+exp(x));
                }
                template <typename T>
                static T backward (T x, T y) {
                    T e = exp(x);
                    return e/(1+e);
                }
            };
            struct tanh {
                static string name () {
                    return "tanh";
                }
                template <typename T>
                static T forward (T x) {
                    return std::tanh(x);
                }
                template <typename T>
                static T backward (T x, T y) {
                    return 1 - y * y;
                }
            };
            struct logistic {
                static string name () {
                    return "logistic";
                }
                template <typename T>
                static T forward (T x) {
                    return 1/(1 + std::exp(-x));
                }
                template <typename T>
                static T backward (T x, T y) {
                    return y * (1 - y);
                }
            };
        }

        template <typename F>
        class FunctionNode: public ArrayNode
        {
            ArrayNode *m_input;
        public:
            FunctionNode (Model *model, Config const &config)
                : ArrayNode(model, config) {
                m_input = findInputAndAdd<ArrayNode>("input", "input");
                resize(*m_input);
                setType(m_input->type());
            }

            void predict () {
                data().apply(m_input->data(), [](Array<>::value_type &y, Array<>::value_type x){y = F::forward(x);});
            }

            void update () {
                m_input->delta().apply(m_input->data(), data(), delta(),
                        [](Array<>::value_type &dx, Array<>::value_type x, Array<>::value_type y, Array<>::value_type dy) {
                            dx += F::backward(x, y) * dy;
                        });
            }
        };

        class ParamNode: public ArrayNode, public role::Params {
            Meta  *m_meta;
            double m_init;
        public:
            ParamNode (Model *model, Config const &config)
                : ArrayNode(model, config),
                  m_meta(findInputAndAdd<Meta>("meta", "meta", "$meta")),
                  m_init(config.get<double>("init", model->config().get<double>("argos.global.init", 0)))
            {
                vector<size_t> size;
                size.push_back(config.get<size_t>("size"));
                resize(size); 
            }

            void sync (Node const *fromNode) {
                ParamNode const *from = dynamic_cast<ParamNode const *>(fromNode);
                BOOST_VERIFY(from);
                data().sync(from->data());
                delta().sync(from->delta());
            }

            void save (ostream &os) const {
                os.write((char const *)this->data().addr(), sizeof(Array<>::value_type) * this->data().size());
                os.write((char const *)this->delta().addr(), sizeof(Array<>::value_type) * this->delta().size());
            }
            void load (istream &is) {
                is.read((char *)this->data().addr(), sizeof(Array<>::value_type) * this->data().size());
                is.read((char *)this->delta().addr(), sizeof(Array<>::value_type) * this->delta().size());
            }

            void init () {
                delta().fill(0);
                if (m_init == 0) {
                    //cerr << "INIT0 " << name() << endl;
                    data().fill(0);
                }
                else {
                    //cerr << "INIT " << name() << endl;
                    std::normal_distribution<Array<>::value_type> normal(0, m_init);
                    Model::Random &random = model()->random();
                    data().apply_serial([&normal, &random](Array<>::value_type &y) {y = normal(random);});
                }
            }

            void predict () {
                if (mode() == MODE_TRAIN) {
                    data().add_scaled(-m_meta->eta(), delta());
                }
            }

            void preupdate () {
                if (mode() == MODE_TRAIN) {
                    if (m_meta->mom() == 0) {   // m_mom has to be 0 for verify mode
                        delta().fill(0);
                    }
                    else {
                        delta().scale(m_meta->mom());
                    }
                    // get norm scale
                    /*
                    double dl2 = delta().l2();
                    if (dl2 == 0) return;
                    double rate = data().l2() / dl2 * m_meta->lambda();
                    */
                    if (m_meta->lambda()) {
                        delta().add_scaled(m_meta->lambda(), data());
                        //delta().add_scaled(rate, data());
                    }
                }
            }
            size_t dim () const {
                return data().size();
            }

            void perturb (size_t index, double epsilon) {
                auto addr = data().addr();
                addr[index] += epsilon;
            }

            double gradient (size_t index) const {
                auto addr = delta().addr();
                return addr[index];
            }

            double value (size_t index) const {
                auto addr = data().addr();
                return addr[index];
            }
        };

        class PadNode: public ArrayNode {
            ArrayNode *m_input;
            size_t m_pad_w, m_pad_h;
            vector<size_t> m_input_shape;
            vector<size_t> m_output_shape;
        public:
            PadNode (Model *model, Config const &config): ArrayNode(model, config) {
                m_input = findInputAndAdd<ArrayNode>("input", "input");
                m_input->data().size(&m_input_shape);
                m_output_shape = m_input_shape;
                setType(m_input->type());
                m_pad_w = config.get<size_t>("width");
                m_pad_h = 0;
                if (type() == IMAGE) {
                    m_pad_h = config.get<size_t>("height");
                    m_output_shape[1] += m_pad_h * 2;
                    m_output_shape[2] += m_pad_w * 2;
                }
                else {
                    m_output_shape[1] += m_pad_w * 2;
                }
                resize(m_output_shape);
                data().fill(0);
                delta().fill(0);
            }

            void predict () {
                if (type() == IMAGE) {
                    Array<>::value_type const *in = m_input->data().addr();
                    for (size_t s = 0; s < m_input_shape[0]; ++s) {
                        Array<>::value_type *out = data().addr();
                        out = data().walk<0>(out, s);
                        out = data().walk<1>(out, m_pad_h);
                        out = data().walk<2>(out, m_pad_w);
                        for (size_t h = 0; h < m_input_shape[1]; ++h) {
                            Array<>::value_type const *in_next = m_input->data().walk<1>(in);
                            copy(in, in_next, out);
                            in = in_next;
                            out = data().walk<1>(out);
                        }
                    }
                }
                else {
                    BOOST_VERIFY(0);
                }
            }

            void update () {
                if (type() == IMAGE) {
                    Array<>::value_type *in = m_input->delta().addr();
                    for (size_t s = 0; s < m_input_shape[0]; ++s) {
                        Array<>::value_type const *out = delta().addr();
                        out = delta().walk<0>(out, s);
                        out = delta().walk<1>(out, m_pad_h);
                        out = delta().walk<2>(out, m_pad_w);
                        for (size_t h = 0; h < m_input_shape[1]; ++h) {
                            Array<>::value_type *in_next = m_input->delta().walk<1>(in);
                            size_t sz = in_next - in;
                            for (size_t o = 0; o < sz; ++o) {
                                in[o] += out[o];
                            }
                            in = in_next;
                            out = delta().walk<1>(out);
                        }
                    }
                    BOOST_VERIFY(in == m_input->delta().addr() + m_input->delta().size());
                }
                else {
                    BOOST_VERIFY(0);
                }
            }
        };

        class LinearNode: public ArrayNode {
            ArrayNode *m_input;
            ParamNode *m_weight;
            ParamNode *m_bias;
            bool m_local;
            size_t m_samples;
            size_t m_rows;  // global: m_rows = m_samples
                            // local: m_rows = m_samples * height [* width]
            size_t m_input_size;
            size_t m_output_size;
        public:
            LinearNode (Model *model, Config const &config)
                : ArrayNode(model, config)
            {
                m_input = findInputAndAdd<ArrayNode>("input", "input");
                if (config.get<int>("local", 0) != 0) {
                    //cerr << "LOCAL " << name() << endl;
                    m_local = true;
                    vector<size_t> size;
                    m_input->data().size(&size);
                    /*
                    cerr << size.size() << ':';
                    for (auto const &v: size) {
                        cerr << " " << v;
                    }
                    cerr << endl;
                    */
                    m_input_size = size.back();
                    m_output_size = config.get<size_t>("channel");
                    BOOST_VERIFY(m_input->data().size() % m_input_size == 0);
                    m_rows = m_input->data().size() / m_input_size;
                    m_samples = size[0];
                    size.back() = m_output_size;
                    resize(size);
                    /*
                    cerr << size.size() << ':';
                    for (auto const &v: size) {
                        cerr << " " << v;
                    }
                    cerr << endl;
                    */
                    //cerr << "ROWS: " << m_rows << endl;
                    setType(m_input->type());
                }
                else {
                    m_local = false;
                    m_samples = m_rows = m_input->data().size(size_t(0));
                    m_input_size = m_input->data().size() / m_samples;
                    m_output_size = config.get<size_t>("channel");
                    vector<size_t> size;
                    size.push_back(m_samples);
                    size.push_back(m_output_size);
                    resize(size);
                }

                m_weight = nullptr;
                try {
                    //m_weight = model->findNode<ParamNode>("weight");
                    m_weight = findInputAndAdd<ParamNode>("weight", "weight");
                }
                catch (...) {
                }
                if (m_weight == nullptr) {
                    Config wconfig = config;
                    wconfig.put("name", name() + "_weight");
                    wconfig.put("type", "param");
                    wconfig.put("size", m_input_size * m_output_size); 
                    wconfig.put("meta", config.get<string>("meta", "$meta"));
                    m_weight = model->createNode<ParamNode>(wconfig);
                    BOOST_VERIFY(m_weight);
                }
                addInput(m_weight, "weight");

                m_bias = nullptr;
                try {
                    m_bias = findInputAndAdd<ParamNode>("bias", "bias");
                }
                catch (...) {
                }
                if (m_bias == nullptr) {
                    Config bconfig = config;
                    bconfig.put("name", name() + "_bias");
                    bconfig.put("type", "param");
                    bconfig.put("size", m_output_size);
                    bconfig.put("init", 0);
                    bconfig.put("meta", config.get<string>("meta", "$meta"));
                    m_bias = model->createNode<ParamNode>(bconfig);
                    BOOST_VERIFY(m_bias);
                }
                addInput(m_bias, "bias");
                BOOST_VERIFY(m_bias->data().size() == m_output_size);
                BOOST_VERIFY(m_weight->data().size() == m_input_size * m_output_size);
            }

            void predict () {
                data().tile(m_bias->data());
                blas::gemm<Array<>::value_type>(m_input->data().addr(), m_rows, m_input_size, false,
                           m_weight->data().addr(), m_input_size, m_output_size, false,
                           this->data().addr(), m_rows, m_output_size, 1.0, 1.0);
            }

            void update () {
                //cerr << "UPDATE " << name() << endl;
                // update input data
                blas::gemm<Array<>::value_type>(this->delta().addr(), m_rows, m_output_size, false,
                           m_weight->data().addr(), m_input_size, m_output_size, true,
                           m_input->delta().addr(), m_rows, m_input_size, 1.0, 1.0);
                // update weight data
                blas::gemm<Array<>::value_type>(m_input->data().addr(), m_rows, m_input_size, true,
                           this->delta().addr(), m_rows, m_output_size, false,
                           m_weight->delta().addr(), m_input_size, m_output_size, 1.0/m_samples, 1.0);
                m_bias->delta().add_scaled_wrapping(1.0/m_samples, delta());
            }
        };

        namespace pool {
            struct max {
                static string name () {
                    return "max";
                }
                typedef unsigned state_type;

                template <typename T>
                static void predict (T const *in, state_type *s, T *out, size_t iw, size_t ow) {
                    unsigned n = iw / ow;
                    copy(in, in + ow, out);
                    fill(s, s + ow, 0);
                    in += ow;
                    for (unsigned i = 1; i < n; ++i) {
                        for (unsigned j = 0; j < ow; ++j) {
                            if (in[j] > out[j]) {
                                out[j] = in[j];
                                s[j] = i;
                            }
                        }
                        in += ow;
                    }
                }

                template <typename T>
                static void update (T *in, state_type const *s, T const *out, size_t iw, size_t ow) {
                    for (unsigned j = 0; j < ow; ++j) {
                        unsigned i = s[j];
                        in[i * ow + j] += out[j];
                    }
                }
            };

            struct avg {
                static string name () {
                    return "avg";
                }
                typedef char state_type;

                template <typename T>
                static void predict (T const *in, state_type *s, T *out, size_t iw, size_t ow) {
                    unsigned n = iw / ow;
                    fill(out, out+ow, 0);
                    for (unsigned i = 0; i < n; ++i) {
                        for (unsigned j = 0; j < ow; ++j) {
                            out[j] += in[j];
                        }
                        in += ow;
                    }
                    for (unsigned j = 0; j < ow; ++j) {
                        out[j] /= n;
                    }
                }

                template <typename T>
                static void update (T *in, state_type const *s, T const *out, size_t iw, size_t ow) {
                    unsigned n = iw / ow;
                    for (unsigned i = 0; i < n; ++i) {
                        for (unsigned j = 0; j < ow; ++j) {
                            in[j] += out[j] / n;
                        }
                        in += ow;
                    }
                }
            };
        }

        template <typename POOL>
        class PoolNode: public ArrayNode {
            ArrayNode *m_input;
            Array<typename POOL::state_type> m_state;
            size_t m_input_channel;
            size_t m_output_channel;
        public:
            PoolNode (Model *model, Config const &config): ArrayNode(model, config) {
                m_input = findInputAndAdd<ArrayNode>("input", "input");
                m_output_channel = config.get<size_t>("channel");
                vector<size_t> size;
                m_input->data().size(&size);
                m_input_channel = size.back();
                size.back() = m_output_channel;
                resize(size);
                m_state.resize(size);
                setType(m_input->type());
            }

            void predict () {
                size_t n = m_input->data().size() / m_input_channel;
#pragma omp parallel for
                for (size_t i = 0; i < n; ++i) {
                    Array<>::value_type const *input = m_input->data().addr() + i * m_input_channel;
                    typename POOL::state_type *state = m_state.addr() + i * m_output_channel;;
                    Array<>::value_type *output = data().addr() + i * m_output_channel;
                    POOL::predict(input, state, output, m_input_channel, m_output_channel);
                }
            }

            void update () {
                size_t samples = m_input->data().size(size_t(0));
#pragma omp parallel for
                for (size_t s = 0; s < samples; ++s) {
                    Array<>::value_type *input = m_input->delta().at(s);
                    typename POOL::state_type const *state = m_state.at(s);
                    Array<>::value_type const *output = delta().at(s);
                    size_t n = m_input->data().size() / m_input_channel / samples;
                    for (size_t i = 0; i < n; ++i) {
                        POOL::update(input, state, output, m_input_channel, m_output_channel);
                        input += m_input_channel;
                        state += m_output_channel;
                        output += m_output_channel;
                    }
                }
            }
        }; 

        class WindowNode: public ArrayNode {
            size_t m_bin;
            size_t m_step;
            size_t m_samples;
            ArrayNode *m_input;
            vector<size_t> m_input_shape;
            vector<size_t> m_output_shape;
        public:
            WindowNode (Model *model, Config const &config): ArrayNode(model, config) {
                m_bin = config.get<size_t>("bin");
                m_step = config.get<size_t>("step");
                m_input = findInputAndAdd<ArrayNode>("input", "input");
                m_input_shape.resize(m_input->data().dim());
                for(size_t i = 0; i < m_input_shape.size(); ++i) {
                    m_input_shape[i] = m_input->data().size(i);
                }
                m_samples = m_input_shape[0];
                BOOST_VERIFY(m_input->type() == IMAGE || m_input->type() == SOUND);

                setType(m_input->type());

                m_output_shape.push_back(m_input_shape[0]);
                m_output_shape.push_back(1 + (m_input_shape[1] - m_bin) / m_step);
                size_t channels = m_input_shape.back() * m_bin;
                if (m_input->type() == IMAGE) {
                    m_output_shape.push_back(1 + (m_input_shape[2] - m_bin) / m_step);
                    setType(IMAGE);
                    channels *= m_bin;
                }
                else {
                    setType(SOUND);
                }
                m_output_shape.push_back(channels);
                resize(m_output_shape);

                /*{
                    vector<size_t> &size = m_output_shape;
                    cerr << name() << ' ';
                    cerr << size.size() << ':';
                    for (auto const &v: size) {
                        cerr << " " << v;
                    }
                    cerr << endl;
                }*/
            }

            void predict () {
                Array<>::value_type const *packed = m_input->data().addr();
                if (type() == IMAGE) {
                    size_t patch_width = m_input->data().walk<2>(packed, m_bin) - packed;
                    //std::cout << patch_width << std::endl;
                    BOOST_VERIFY(patch_width == m_output_shape.back() / m_bin);
#pragma omp parallel for
                    for (size_t i = 0; i < m_samples; ++i) {
                        Array<>::value_type const *packed_1 = m_input->data().walk<0>(packed, i);
                        Array<>::value_type *unpacked = data().walk<0>(data().addr(), i);
                        for (size_t j = 0; j + m_bin <= m_input_shape[1]; j += m_step) { // row
                            Array<>::value_type const *packed_2 = packed_1; // TODO??? maybe start from an offset
                            for (size_t k = 0; k + m_bin <= m_input_shape[2]; k += m_step) { // col
                                ////
                                Array<>::value_type const *packed_3 = packed_2; // TODO??? maybe start from an offset
                                for (size_t l = 0; l < m_bin; ++l) {    // m_bin rows
                                    copy(packed_3, packed_3 + patch_width, unpacked);
                                    unpacked += patch_width;
                                    packed_3 = m_input->data().walk<1>(packed_3);
                                }
                                packed_2 = m_input->data().walk<2>(packed_2, m_step);
                            }
                            packed_1 = m_input->data().walk<1>(packed_1, m_step);
                        }
                    }
                }
                else {
                    /*
                    for (size_t i = 0; i < m_samples; ++i) {
                        Array<>::value_type *unpacked = data().at(i);
                        Array<>::value_type const *packed = m_input->data().at(i);
                        for (size_t j = 0; j < m_input_shape[1]; j += m_bin;) {
                            unpacked += m_output_shape.back();
                        }
                    }
                    */
                    BOOST_VERIFY(0);
                }
            }

            void update () {
                Array<>::value_type *packed = m_input->delta().addr();
                if (type() == IMAGE) {
                    size_t patch_width = m_input->data().walk<2>(packed, m_bin) - packed;
                    //std::cout << patch_width << std::endl;
                    BOOST_VERIFY(patch_width == m_output_shape.back() / m_bin);
#pragma omp parallel for
                    for (size_t i = 0; i < m_samples; ++i) {
                        Array<>::value_type *packed_1 = m_input->data().walk<0>(packed, i);
                        Array<>::value_type const *unpacked = delta().walk<0>(delta().addr(), i);
                        for (size_t j = 0; j + m_bin <= m_input_shape[1]; j += m_step) { // row
                            Array<>::value_type *packed_2 = packed_1; // TODO??? maybe start from an offset
                            for (size_t k = 0; k + m_bin <= m_input_shape[2]; k += m_step) { // col
                                ////
                                Array<>::value_type *packed_3 = packed_2; // TODO??? maybe start from an offset
                                for (size_t l = 0; l < m_bin; ++l) {    // m_bin rows
                                    //copy(packed_3, packed_3 + patch_width, unpacked);
                                    for (unsigned m = 0; m < patch_width; ++m) {
                                        packed_3[m] += unpacked[m];
                                    }
                                    unpacked += patch_width;
                                    packed_3 = m_input->data().walk<1>(packed_3);
                                }
                                packed_2 = m_input->data().walk<2>(packed_2, m_step);
                            }
                            packed_1 = m_input->data().walk<1>(packed_1, m_step);
                        }
                    }
                }
                else {
                    /*
                    for (size_t i = 0; i < m_samples; ++i) {
                        Array<>::value_type *unpacked = data().at(i);
                        Array<>::value_type const *packed = m_input->data().at(i);
                        for (size_t j = 0; j < m_input_shape[1]; j += m_bin;) {
                            unpacked += m_output_shape.back();
                        }
                    }
                    */
                    BOOST_VERIFY(0);
                }
            }
        };

        class SoftMaxNode: public ArrayNode
        {
            ArrayNode *m_input;
        public:
            SoftMaxNode (Model *model, Config const &config)
                : ArrayNode(model, config) {
                m_input = findInputAndAdd<ArrayNode>("input", "input");
                resize(*m_input);
                setType(m_input->type());
            }

            void predict () {
                size_t samples = m_input->data().size(size_t(0));
                size_t sz = m_input->data().size() / samples;
                Array<>::value_type const *in = m_input->data().addr();
                Array<>::value_type *out = data().addr();
                vector<Array<>::value_type> e(sz);
                for (size_t i = 0; i < samples; ++i) {
                    Array<>::value_type sum = 0;
                    for (size_t j = 0; j < sz; ++j) {
                        e[j] = exp(in[j]);
                        sum += e[j];
                    }
                    for (size_t j = 0; j < sz; ++j) {
                        out[j] = e[j]/sum;
                    }
                    in += sz;
                    out += sz;
                }
            }

            void update () {
                size_t samples = m_input->data().size(size_t(0));
                size_t sz = m_input->data().size() / samples;
                Array<>::value_type const *in = m_input->data().addr();
                Array<>::value_type *in_delta = m_input->delta().addr();
                Array<>::value_type const *out = data().addr();
                Array<>::value_type const *out_delta = delta().addr();

                //cerr << "SOFTMAX UPDATE" << endl;
                for (;;) {
                    if (outputs().size() != 1) break;
                    Node *node = outputs()[0].node;
                    LogPOutputNode *logp = dynamic_cast<LogPOutputNode *>(node);
                    if (logp == nullptr) break;
                    //cerr << "OPTIMIZE SOFTMAX + LOGP" << endl;
                    vector<int> const &labels = logp->inputLabels();
                    BOOST_VERIFY(labels.size() == samples);
                    for (size_t i = 0; i < samples; ++i) {
                        for (size_t j = 0; j < sz; ++j) {
                            in_delta[j] += out[j];
                        }
                        int l = labels[i];
                        BOOST_VERIFY(l < int(sz));
                        BOOST_VERIFY(l >= 0);
                        if (l < int(sz)) {
                            in_delta[l] -= 1.0;
                        }
                        in_delta += sz;
                        out += sz;
                    }
                    return;
                }

                for (size_t i = 0; i < samples; ++i) {
                    Array<>::value_type sum = 0;
                    for (unsigned j = 0; j < sz; ++j) {
                        sum += out_delta[j] * out[j];
                    }
                    for (unsigned j = 0; j < sz; ++j) {
                        in_delta[j] += out[j] * (out_delta[j] - sum);
                    }
                    in += sz;
                    in_delta += sz;
                    out += sz;
                    out_delta += sz;
                }
            }
        };

        class NormalizeNode: public ArrayNode
        {
            ArrayNode *m_input;
            vector<Array<>::value_type> m_rate;
            size_t m_samples;
            size_t m_dim;
        public:
            NormalizeNode (Model *model, Config const &config)
                : ArrayNode(model, config) {
                m_input = findInputAndAdd<ArrayNode>("input", "input");
                resize(*m_input);
                setType(m_input->type());
                m_samples = data().size(size_t(0));
                m_dim = data().size() / m_samples;
                m_rate.resize(m_samples);
            }

            void predict () {
                Array<>::value_type const *in = m_input->data().addr();
                Array<>::value_type *out = data().addr();
                for (size_t i = 0; i < m_samples; ++i) {
                    Array<>::value_type sum = 0;
                    for (size_t j = 0; j < m_dim; ++j) {
                        sum += in[j] * in[j];;
                    }
                    Array<>::value_type r = 1.0 / sqrt(sum / m_dim);
                    m_rate[i] = r;
                    for (size_t j = 0; j < m_dim; ++j) {
                        out[j] = in[j] * r;
                    }
                    in += m_dim;
                    out += m_dim;
                }
            }

            void update () {
                Array<>::value_type const *in = m_input->data().addr();
                Array<>::value_type *in_delta = m_input->delta().addr();
                Array<>::value_type const *out = data().addr();
                Array<>::value_type const *out_delta = delta().addr();
                for (size_t i = 0; i < m_samples; ++i) {
                    for (size_t j = 0; j < m_dim; ++j) {
                        in_delta[j] = out_delta[j] * m_rate[i] * (1.0 - out[j] * out[j] / m_dim);
                    }

                    in += m_dim;
                    out += m_dim;
                    in_delta += m_dim;
                    out_delta += m_dim;
                }
            }
        };

        class DropOutNode: public ArrayNode
        {
            ArrayNode *m_input;
            double m_rate;
            int m_freq;
            vector<Array<>::value_type> m_mask;
            size_t m_samples;
            size_t m_sample_size;
            size_t m_cnt;
        public:
            DropOutNode (Model *model, Config const &config)
                : ArrayNode(model, config) {
                m_input = findInputAndAdd<ArrayNode>("input", "input");
                m_rate = config.get<double>("rate", 0.5);
                m_freq = config.get<double>("freq", 1);
                m_cnt = 0;
                resize(*m_input);
                setType(m_input->type());
                m_samples = data().size(size_t(0));
                m_sample_size = data().size() / m_samples;
                m_mask.resize(m_sample_size, 0);
                size_t nz = m_mask.size() * m_rate;
                m_rate = double(nz) / m_mask.size();
                for (size_t i = 0; i < nz; ++i) {
                    m_mask[i] = 1.0;
                }
            }

            void predict () {
                if (mode() == MODE_PREDICT) {
#pragma omp parallel for
                    for (size_t i = 0; i < m_samples; ++i) {
                        Array<>::value_type const *in = m_input->data().at(i);
                        Array<>::value_type *out = data().at(i);
                        for (size_t j = 0; j < m_sample_size; ++j) {
                            out[j] = in[j] * m_rate;
                        }
                    }
                }
                else {
                    if (m_cnt % m_freq == 0) {
                        random_shuffle(m_mask.begin(), m_mask.end());
                    }
                    ++m_cnt;
#pragma omp parallel for
                    for (size_t i = 0; i < m_samples; ++i) {
                        Array<>::value_type const *in = m_input->data().at(i);
                        Array<>::value_type *out = data().at(i);
                        for (size_t j = 0; j < m_sample_size; ++j) {
                            out[j] = m_mask[j] * in[j];
                        }
                    }
                }
                //data().apply(m_input->data(), [](Array<>::value_type &y, Array<>::value_type x){y = F::forward(x);});
            }

            void update () {
#pragma omp parallel for
                for (size_t i = 0; i < m_samples; ++i) {
                    Array<>::value_type *in = m_input->delta().at(i);
                    Array<>::value_type const *out = delta().at(i);
                    for (size_t j = 0; j < m_sample_size; ++j) {
                        in[j] += m_mask[j] * out[j];
                    }
                }
            }
        };
    }
}
#endif

