#ifndef ARGOS_IO
#define ARGOS_IO
#include <fstream>
#include <sstream>
#include <algorithm>

namespace argos {

    namespace neural {

        using namespace std;

        class LabelTap: public Node {
            LabelOutputNode *m_input;
            string path;
            ofstream m_file;
        public:
            LabelTap (Model *model, Config const &config): Node(model, config),
                path(config.get<string>("path"))
            {
                m_input = findInputAndAdd<LabelOutputNode>("input", "input");
                BOOST_VERIFY(m_input);
            }

            void predict (Mode mode) {
                if (mode == MODE_PREDICT) {
                    if (!m_file.is_open()) {
                        m_file.open(path);
                    }
                    vector<int> const &labels = m_input->labels();
                    for (int l: labels) {
                        m_file << l << endl;
                    }
                }
            }
        };

        class SimpleArrayInputNode: public ArrayNode, public LabelInput, public Input {

            class Samples {
                size_t m_dim;
                vector<int> m_labels;
                vector<vector<pair<unsigned, double>>> m_data;
                vector<unsigned> m_index;
                bool m_test;
                size_t m_off;
            public:
                Samples (string const &path, bool is_test = false): m_test(is_test), m_off(0) {
                    cerr << "Loading " << path << "..." << endl;
                    m_data.clear();
                    m_labels.clear();
                    ifstream is(path.c_str());
                    string line;
                    m_dim = 0;
                    while (getline(is, line)) {
                        istringstream ss(line);
                        unsigned l, d;
                        char dummy;
                        double v;
                        ss >> l;
                        if (!ss) continue;
                        m_labels.push_back(l);
                        m_data.push_back(vector<pair<unsigned, double>>());
                        auto &back = m_data.back();
                        while (ss >> d >> dummy >> v) {
                            BOOST_VERIFY(dummy == ':');
                            back.push_back(make_pair(d - 1, v));
                            if (d > m_dim) m_dim = d;
                        }
                    }
                    m_index.resize(m_labels.size());
                    for (unsigned i = 0; i < m_index.size(); ++i) m_index[i] = i;
                    if (!is_test) {
                        random_shuffle(m_index.begin(), m_index.end());
                    }
                }

                size_t dim () const {
                    return m_dim;
                }

                void reset (void) {
                    m_off = 0;
                }

                void fill (unsigned batch, unsigned dim, Array<> *data, vector<int> *labels) {
                    labels->resize(batch);
                    if (m_off + batch >= m_index.size()) {
                        if (!m_test) {
                            random_shuffle(m_index.begin(), m_index.end());
                            m_off = 0;
                        }
                    }
                    if (m_off >= m_index.size()) {
                        throw StopIterationException();
                    }
                    data->fill(0.0);
                    std::fill(labels->begin(), labels->end(), -1);
                    Array<>::value_type *x = data->addr();
                    for (unsigned b = 0; b < batch; ++b) {
                        if (m_off >= m_index.size()) {
                            labels->resize(b);
                            break;
                        }
                        size_t i = m_index[m_off++];
                        labels->at(b) = m_labels[i];
                        for (auto p: m_data[i]) {
                            x[p.first] = p.second;
                        }
                        x += dim;
                    }
                }
            };

            size_t m_batch;
            size_t m_dim;
            Samples m_train;
            Samples m_test;
            vector<int> m_labels;
        public:
            SimpleArrayInputNode (Model *model, Config const &config)
                : ArrayNode(model, config),
                  m_batch(config.get<size_t>("batch")), //, model->config().get<size_t>("argos.default.batch"))),
                  m_train(config.get<string>("train")), //, model->config().get<string>("argos.default.train")), m_dim),
                  m_test(config.get<string>("test"), true) //, model->config().get<string>("argos.default.test")), m_dim, true)
            {
                m_dim = m_train.dim();
                if (m_test.dim() > m_dim) m_dim = m_test.dim();
                vector<size_t> size{m_batch, m_dim};
                resize(size);
                m_labels.resize(m_batch);
            }

            void reset (Mode mode) {
                if (mode == MODE_PREDICT) {
                    m_test.reset();
                }
                else {
                    BOOST_VERIFY(0);    // for now don't support reset for training
                }
            }

            void predict (Mode mode) {
                if (mode == MODE_PREDICT) {
                    m_test.fill(m_batch, m_dim, &data(), &m_labels);
                }
                else {
                    m_train.fill(m_batch, m_dim, &data(), &m_labels);
                }
            }

            virtual vector<int> const &labels () const {
                return m_labels;
            }
        };
    }
}
#endif
