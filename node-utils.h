#ifndef ARGOS_NODE_UTILS
#define ARGOS_NODE_UTILS
#include <fstream>
#include <sstream>
#include <algorithm>

namespace argos {
    namespace utils {
        using namespace std;
        using namespace argos::core;

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

            void predict () {
                if (mode() == MODE_PREDICT) {
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

        class LibSvmInputNode: public ArrayNode, public role::LabelInput, public role::Input {
            size_t m_batch;
            size_t m_dim;
            size_t m_off;
            vector<int> m_labels;
            vector<vector<pair<unsigned, double>>> m_data;
            vector<unsigned> m_index;
            vector<int> m_batch_labels;
        public:
            LibSvmInputNode (Model *model, Config const &config)
                : ArrayNode(model, config),
                  m_batch(config.get<size_t>("batch")),
                  m_dim(config.get<unsigned>("dim")),
                  m_off(0)
            {
                LOG(debug) << "dim: " << m_dim;
                LOG(debug) << "batch: " << m_batch;
                vector<size_t> size{m_batch, m_dim};
                resize(size);

                string path;
                if (mode() == MODE_PREDICT) {
                    path = config.get<string>("test");
                }
                else {
                    path = config.get<string>("train");
                }

                cerr << "Loading " << path << "..." << endl;
                ifstream is(path.c_str());
                string line;
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
                        BOOST_VERIFY(d <= m_dim);
                    }
                }
                m_index.resize(m_labels.size());
                for (unsigned i = 0; i < m_index.size(); ++i) m_index[i] = i;
                if (mode() != MODE_PREDICT) {
                    random_shuffle(m_index.begin(), m_index.end());
                }
            }

            void rewind () {
                m_off = 0;
            }

            void predict () {
                m_batch_labels.resize(m_batch);
                if (m_off + m_batch > m_index.size()) {
                    if (mode() == MODE_TRAIN) {
                        random_shuffle(m_index.begin(), m_index.end());
                        m_off = 0;
                        BOOST_VERIFY(m_off + m_batch <= m_index.size());
                    }
                }
                if (m_off >= m_index.size()) {
                    throw StopIterationException();
                }
                data().fill(0.0);
                Array<>::value_type *x = data().addr();
                for (unsigned b = 0; b < m_batch; ++b) {
                    if (m_off >= m_index.size()) {
                        m_batch_labels.resize(b);
                        break;
                    }
                    size_t i = m_index[m_off++];
                    m_batch_labels[b] = m_labels[i];
                    for (auto p: m_data[i]) {
                        if (p.first >= m_dim) {
                            LOG(error) << "dim exceeds range: " << p.first;
                        }
                        x[p.first] = p.second;
                    }
                    x += m_dim;
                }
            }

            virtual vector<int> const &labels () const {
                return m_batch_labels;
            }
        };

    /// The node evaluates the model periodically in training mode.
        class Eval: public Node {
            /// Clone the model for prediction.
            unique_ptr<Model> m_modelClone;
            unsigned m_period;
            unsigned m_loop;
        public:
            Eval (Model *model, Config const &config)
                : Node(model, config),
                  m_modelClone(nullptr),
                  m_period(config.get<unsigned>("period", 100)),
                  m_loop(0)
            {
                findInputAndAdd<Node>("input", "input");
                if ((mode() == MODE_TRAIN) && (m_period > 0)) {
                    m_modelClone = unique_ptr<Model>(new Model(*model, MODE_PREDICT)); 
                }
            }
            void predict () {
                ++m_loop;
                if ((mode() == MODE_TRAIN) && (m_period > 0) && (m_loop % m_period == 0)) {
                    m_modelClone->sync(*model());
                    m_modelClone->predict();
                    cout << "EVAL ";
                    m_modelClone->report(cout, true);
                }
            }
        };
    }
}
#endif
