#ifndef ARGOS_NODE_DREAM
#define ARGOS_NODE_DREAM

namespace argos {

    namespace dream {

        double spearman (vector<int> const &r1, vector<int> &r2) {
            BOOST_VERIFY(r1.size() == r2.size());
            double z = 0;
            for (unsigned j = 0; j < r1.size(); ++j) {
                int d = r1[j] - r2[j];
                z += d * d;
            }
            return 1.0 - 6 * z / r1.size() / (r1.size() * r1.size() - 1);
        }

        bool LoadFile (std::string const &path,
                Array<> *output,
                vector<string> *row_names,
                vector<string> *col_names) {
            ifstream is(path.c_str());
            if (!is) return false;
            string dummy;
            getline(is, dummy);
            size_t rows, cols;
            is >> cols >> rows;
            BOOST_VERIFY(is);
            output->resize(2, rows, cols);  // input file is transposed
            row_names->resize(rows);
            col_names->resize(cols);
            // read header
            is >> dummy >> dummy;
            for (unsigned i = 0; i < rows; ++i) {
                is >> row_names->at(i);
            }
            for (unsigned i = 0; i < cols; ++i) {
                is >> col_names->at(i) >> dummy;
                for (unsigned j = 0; j < rows; ++j) {
                    is >> output->at(j)[i];
                }
            }
            BOOST_VERIFY(is);
            return true;
        }

        void SaveFile (std::string const &path,
                Array<> const &input,
                vector<string> const &row_names,
                vector<string> const &col_names) {
            ofstream os(path.c_str());
            vector<size_t> sz;
            input.size(&sz);
            os << "#1.2" << endl;
            os << sz[1] << '\t' << sz[0] << endl;
            os << "Name\tDescription";
            for (auto const &s: row_names) {
                os << '\t' << s;
            }
            os << endl;
            for (unsigned i = 0; i < sz[1]; ++i) {
                os << col_names[i] << '\t' << col_names[i];
                for (unsigned j = 0; j < sz[0]; ++j) {
                    os << '\t' << input.at(j)[i];
                }
                os << endl;
            }
        }

        class DataNode: public core::ArrayNode, public role::Input, public role::ArrayLabelInput {
            vector<string> m_rows;          // cell lines
            vector<string> m_cols_exp;      // expression feature
            vector<string> m_cols_copy;      // copy number feature
            vector<string> m_cols_target;      // target functions
            string m_dir;
            Array<> m_exp;
            Array<> m_copy;
            Array<> m_target;
            bool m_done;
        public:
            DataNode (Model *model, Config const &config) 
                : ArrayNode(model, config), m_done(false)
            {
                string const &dir = m_dir = config.get<string>("dir", ".");
                LOG(info) << "loading data from " << dir;
                vector<string> tmp; // string for row name check
                if (mode() == MODE_PREDICT) {
                    LoadFile(dir + "/test.expression", &m_exp, &m_rows, &m_cols_exp);
                    LoadFile(dir + "/test.copy", &m_copy, &tmp, &m_cols_copy);
                    BOOST_VERIFY(m_rows == tmp);
                    // we need to load the training target to figout out the
                    // output columns
                    if (!LoadFile(dir + "/test.target", &m_target, &tmp, &m_cols_target)) {
                        LoadFile(dir + "/train.target", &m_target, &tmp, &m_cols_target);
                        vector<size_t> sz;
                        m_target.size(&sz);
                        sz[0] = m_exp.size(size_t(0)); // resize # rows to testing number
                        m_target.resize(sz);
                        m_target.fill(0);
                    }
                }
                else {
                    LoadFile(dir + "/train.expression", &m_exp, &m_rows, &m_cols_exp);
                    LoadFile(dir + "/train.copy", &m_copy, &tmp, &m_cols_copy);
                    BOOST_VERIFY(m_rows == tmp);
                    LoadFile(dir + "/train.target", &m_target, &tmp, &m_cols_target);
                    BOOST_VERIFY(m_rows == tmp);
                }
                //unsigned batch = getConfig<unsigned>("batch", "argos.global.batch");
                //role::BatchInput::init(batch, m_rows.size(), mode());
                vector<size_t> sz{m_rows.size(), m_cols_exp.size() + m_cols_copy.size()};
                data().resize(sz);
                delta().resize(sz);
                delta().fill(0.0);
                for (unsigned i = 0; i < m_rows.size(); ++i) {
                    Array<>::value_type *x = data().at(i);
                    Array<>::value_type const *x_exp = m_exp.at(i);
                    Array<>::value_type const *x_copy = m_copy.at(i);
                    unsigned j = 0;
                    for (unsigned l = 0; l < m_cols_exp.size(); ++l) {
                        x[j++] = x_exp[l];
                    }
                    for (unsigned l = 0; l < m_cols_copy.size(); ++l) {
                        x[j++] = x_copy[l];
                    }
                    BOOST_VERIFY(j == sz[1]);
                }
            }

            Array<> const &target () const {
                return m_target;
            }

            vector<string> const &rowNames () const {
                return m_rows;
            }

            vector<string> const &targetColNames () const {
                return m_cols_target;
            }

            string const &dir () const {
                return m_dir;
            }

            virtual void rewind () {
                m_done = false;
            }

            void predict () {
                if (mode() == MODE_PREDICT) {
                   if (m_done) throw StopIterationException();
                    m_done = true;
                }
            }

            virtual Array<double> const &labels () const {
                return m_target;
            }

            virtual void report () const {
                ArrayNode::report();
                cerr << "target:\t" << m_target.l2() << endl;
            }
        };

        class RankCorrelationNode: public Node, public role::Stat {
            core::ArrayNode *m_input;
            DataNode *m_data;
        public:
            RankCorrelationNode (Model *model, Config const &config) 
                : Node(model, config),
                  m_input(findInputAndAdd<core::ArrayNode>("input", "input")),
                  m_data(findInputAndAdd<DataNode>("label", "label"))
            {
                role::Stat::init({"spearman"});
            }

            void predict () {
                Array<> const &pred = m_input->data();
                Array<> const &truth = m_data->target();
                vector<size_t> sz1, sz2;
                pred.size(&sz1); truth.size(&sz2);
                BOOST_VERIFY(sz1 == sz2);
                unsigned cells = sz1[0];
                unsigned genes = sz1[1];
                for (unsigned i = 0; i < genes; ++i) {
                    vector<int> rank1(cells);
                    vector<int> rank2(cells);
                    vector<pair<float, unsigned>> score1(cells);
                    vector<pair<float, unsigned>> score2(cells);
                    for (unsigned j = 0; j < cells; ++j) {
                        score1[j] = make_pair(pred.at(j)[i], j);
                        score2[j] = make_pair(truth.at(j)[i], j);
                    }
                    sort(score1.begin(), score1.end());
                    sort(score2.begin(), score2.end());
                    for (unsigned j = 0; j < cells; ++j) {
                        rank1[score1[j].second] = j + 1;
                        rank2[score2[j].second] = j + 1;
                    }
                    acc(0)(spearman(rank1, rank2));
                }
            }

            void update () {
            }
        };

        class OutputTap: public Node {
            core::ArrayNode const *m_input;
            DataNode const *m_data;
        public:
            OutputTap (Model *model, Config const &config): Node(model, config),
                m_input(findInputAndAdd<core::ArrayNode>("input", "input")),
                m_data(findInputAndAdd<DataNode>("data", "data"))
            {
                BOOST_VERIFY(m_input);
                BOOST_VERIFY(m_data);
            }

            void predict () {
                if (mode() == MODE_PREDICT) {
                    Array<> const &array = m_input->data();
                    SaveFile(m_data->dir() + "/test.output", array, m_data->rowNames(), m_data->targetColNames());
                }
            }
        };
    }
}

#endif

