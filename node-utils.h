#ifndef ARGOS_NODE_UTILS
#define ARGOS_NODE_UTILS
#include <fstream>
#include <sstream>
#include <algorithm>

namespace argos {
    namespace utils {
        using namespace std;
        using namespace argos::core;

        template <typename T>
        class LabelTap: public Node {
            LabelOutputNode<T> *m_input;
            string path;
            ofstream m_file;
        public:
            LabelTap (Model *model, Config const &config): Node(model, config),
                path(config.get<string>("path"))
            {
                m_input = findInputAndAdd<LabelOutputNode<T>>("input", "input");
                BOOST_VERIFY(m_input);
            }

            void predict () {
                if (mode() == MODE_PREDICT) {
                    if (!m_file.is_open()) {
                        m_file.open(path);
                    }
                    vector<T> const &labels = m_input->labels();
                    for (T l: labels) {
                        m_file << l << endl;
                    }
                }
            }
        };

        template <typename T = int>
        class LibSvmInputNode: public ArrayNode, public role::LabelInput<T>, public role::BatchInput {
            size_t m_dim;
            vector<T> m_all_labels;
            vector<vector<pair<unsigned, double>>> m_data;
            vector<unsigned> m_index;
            vector<T> m_labels;
        public:
            LibSvmInputNode (Model *model, Config const &config)
                : ArrayNode(model, config),
                m_dim(config.get<unsigned>("dim"))
            {
                string path;
                if (mode() == MODE_PREDICT) {
                    path = config.get<string>("test");
                }
                else {
                    path = config.get<string>("train");
                }

                LOG(info) << "loading " << path;
                ifstream is(path.c_str());
                string line;
                while (getline(is, line)) {
                    istringstream ss(line);
                    T l;
                    unsigned d;
                    char dummy;
                    double v;
                    ss >> l;
                    if (!ss) continue;
                    m_all_labels.push_back(l);
                    m_data.push_back(vector<pair<unsigned, double>>());
                    auto &back = m_data.back();
                    while (ss >> d >> dummy >> v) {
                        BOOST_VERIFY(dummy == ':');
                        back.push_back(make_pair(d - 1, v));
                        BOOST_VERIFY(d <= m_dim);
                    }
                }
                role::BatchInput::init(getConfig<unsigned>("batch", "argos.global.batch"), m_all_labels.size(), mode());
                LOG(debug) << "dim: " << m_dim;
                vector<size_t> size{batch(), m_dim};
                resize(size);
                setType(FLAT);
            }

            void predict () {
                data().fill(0.0);
                m_labels.clear(); //resize(m_batch);
                Array<>::value_type *x = data().addr();
                role::BatchInput::next([this, &x](unsigned i) {
                    m_labels.push_back(m_all_labels[i]);
                    for (auto p: m_data[i]) {
                        if (p.first >= m_dim) {
                            LOG(error) << "dim exceeds range: " << p.first;
                        }
                        x[p.first] = p.second;
                    }
                    x += m_dim;
                });
            }

            virtual vector<T> const &labels () const {
                return m_labels;
            }
        };

        /// The node evaluates the model periodically in training mode.
        // Be careful! The Eval node in the copied mode will also be run -- in
        // PREDICT mode.  So PREDICT mode must not do anything, or it will
        // over-write existing data.
        class Eval: public Node {
            /// Clone the model for prediction.
            ofstream os;
            string m_report_node;
            unsigned m_period;
            unsigned m_loop;
            Node *m_root;
        public:
            Eval (Model *model, Config const &config)
                : Node(model, config),
                  m_report_node(config.get<string>("node", "")),
                  m_period(config.get<unsigned>("period", 100)),
                  m_loop(0)
            {
                m_root = model->findNode<Node>(config.get<string>("root"));
                BOOST_VERIFY(m_root);
                string path = config.get<string>("output", "");
                if (mode() == MODE_TRAIN) {
                    if (path.size()) {
                        os.open(path.c_str());
                    }
                }
            }

            ~Eval () {
                if (mode() == MODE_TRAIN) {
                    if (os.is_open()) {
                        os.close();
                    }
                }
            }

            void prepare (Plan *plan) {
                if (mode() == MODE_TRAIN) {
                    // add preupdate task
                    auto deps = plan->add(this, TASK_UPDATE, std::bind(&Node::update, dynamic_cast<Node *>(this)));
                    deps.add(m_root, TASK_UPDATE);
                }
            }

            void update () {
                ++m_loop;
                if ((mode() == MODE_TRAIN) && (m_period > 0) && (m_loop % m_period == 0)) {
                    unique_ptr<Model> m_modelClone(new Model(*model(), MODE_PREDICT)); 
                    m_modelClone->sync(*model());
                    m_modelClone->predict();
                    if (m_report_node.size()) {
                        Node *node = m_modelClone->findNode<Node>(m_report_node);
                        BOOST_VERIFY(node);
                        if (os.is_open()) {
                            node->report(os);
                            os.flush();
                        }
                        else {
                            node->report(cout);
                        }
                    }
                    else {
                        if (os.is_open()) {
                            m_modelClone->report(os, true);
                            os.flush();
                        }
                        else {
                            m_modelClone->report(cout, true);
                        }
                    }
                }
            }
        };

        class ArrayStat: public Node, public role::Stat {
            ArrayNode *m_input;
        public:
            ArrayStat (Model *model, Config const &config)
                : Node(model, config),
                  m_input(findInputAndAdd<ArrayNode>("input", "input"))
            {
                  role::Stat::init({"data", "delta"});
            }

            void doStat () {
                Stat::reset();
                auto &acc0 = acc(0);
                auto &acc1 = acc(1);
                Array<>::value_type const *x = m_input->data().addr();
                Array<>::value_type *dx = m_input->delta().addr();
                size_t sz = m_input->data().size();
                BOOST_VERIFY(sz == m_input->delta().size());
                for (size_t i = 0; i < sz; ++i) {
                    acc0(x[i]);
                    acc1(dx[i]);
                }
            }

            void prepare (Plan *plan) {
                Node::prepare(plan);
                auto r = plan->add(this, TASK_USER, std::bind(&ArrayStat::doStat, this));
                switch (mode()) {
                    case MODE_TRAIN:
                        r.add(m_input, TASK_UPDATE);
                    case MODE_PREDICT:
                        r.add(m_input, TASK_PREDICT);
                    }
                }
            };
    }
}
#endif
