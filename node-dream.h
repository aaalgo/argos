#ifndef ARGOS_NODE_DREAM
#define ARGOS_NODE_DREAM

#include <iomanip>
#include <limits>
#include <thread>

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

        class DataNode: public core::ArrayNode, public role::BatchInput, public role::ArrayLabelInput {
            vector<string> m_rows;          // cell lines
            vector<string> m_cols_exp;      // expression feature
            vector<string> m_cols_copy;      // copy number feature
            vector<string> m_cols_target;      // target functions
            string m_dir;
            Array<> m_exp;
            Array<> m_copy;
            Array<> m_target;
            Array<> m_margins;
            Array<> m_all_margins;
            Array<> m_all_target;           // for training
            bool m_done;
            bool m_do_exp;
            bool m_do_copy;

            Model::Random m_random;
            double m_noise_level;
            Array<> m_noise;
            future<void> m_noise_task;
            double m_margin;

            void fill_noise () {
                std::normal_distribution<Array<>::value_type> normal(0, 1.0);
                size_t row = m_noise.size(size_t(0));
                size_t col = m_noise.size(size_t(1));
                for (unsigned i = 0; i < row; ++i) {
                    Array<>::value_type *x = m_noise.at(i);
                    double sum = 0;
                    for (unsigned j = 0; j < col; ++j) {
                        double a = x[j] = normal(m_random);
                        sum += a * a;
                    }
                    double r = m_noise_level / sqrt(sum);
                    for (unsigned j = 0; j < col; ++j) {
                        x[j] *= r;
                    }
                }
            }

        public:
            DataNode (Model *model, Config const &config) 
                : ArrayNode(model, config), m_done(false),
                m_noise_level(config.get<double>("noise", 0)),
                m_margin(config.get<double>("margin", 0)
                
            {
                unsigned mask = config.get<unsigned>("mask", 1);
                bool logscale = (config.get<unsigned>("logscale", 0) != 0);
                if (logscale) {
                    cerr << "Applying logscale to data." << endl;
                }
                m_do_exp = (mask & 1) ? 1 : 0;
                m_do_copy = (mask & 2) ? 1 : 0;
                BOOST_VERIFY(mask > 0);
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
                    {
                        vector<size_t> sz;
                        m_target.size(&sz);
                        sz.push_back(2);
                        m_margins.resize(sz);
                        m_margins.fill(0);
                    }


                    vector<size_t> sz{m_rows.size(), m_do_exp * m_cols_exp.size() + m_do_copy * m_cols_copy.size()};
                    data().resize(sz);
                    delta().resize(sz);
                    delta().fill(0.0);
                    for (unsigned i = 0; i < m_rows.size(); ++i) {
                        Array<>::value_type *x = data().at(i);
                        unsigned j = 0;
                        if (m_do_exp) {
                            Array<>::value_type const *x_exp = m_exp.at(i);
                            for (unsigned l = 0; l < m_cols_exp.size(); ++l) {
                                x[j++] = x_exp[l];
                            }
                        }
                        if (m_do_copy) {
                            Array<>::value_type const *x_copy = m_copy.at(i);
                            for (unsigned l = 0; l < m_cols_copy.size(); ++l) {
                                x[j++] = x_copy[l];
                            }
                        }
                        BOOST_VERIFY(j == sz[1]);
                    }

                    if (logscale) {
                        BOOST_VERIFY(!m_do_copy);
                        data().apply([](double &y){y = (std::log(y) - 1.0) / 1.75;});
                    }
                }
                else {
                    LoadFile(dir + "/train.expression", &m_exp, &m_rows, &m_cols_exp);
                    LoadFile(dir + "/train.copy", &m_copy, &tmp, &m_cols_copy);
                    BOOST_VERIFY(m_rows == tmp);
                    LoadFile(dir + "/train.target", &m_all_target, &tmp, &m_cols_target);
                    BOOST_VERIFY(m_rows == tmp);

                    if (logscale) {
                        BOOST_VERIFY(!m_do_copy);
                        m_exp.apply([](double &y){y = (std::log(y) - 1.0) / 1.75;});
                    }

                    unsigned batch = getConfig<unsigned>("batch", "argos.global.batch");
                    role::BatchInput::init(batch, m_rows.size(), mode());

                    vector<size_t> sz{batch, m_do_exp * m_cols_exp.size() + m_do_copy * m_cols_copy.size()};
                    data().resize(sz);
                    delta().resize(sz);
                    if (m_noise_level > 0) {
                        m_noise.resize(sz);
                        m_noise_task = std::async(std::launch::async, [this](){this->fill_noise();});
                    }

                    m_all_target.size(&sz);
                    sz[0] = batch;
                    m_target.resize(sz);

                    sz.push_back(2);
                    m_margins.resize(sz);
                    sz[0] = m_rows.size();
                    m_all_margins.resize(sz);


                    // compute all margins
                    size_t dim = sz[1];
                    for (unsigned i = 0; i < m_rows.size(); ++i) {
                        Array<>::value_type *y = m_all_target.at(i);
                        Array<>::value_type *m = m_all_margins.at(i);
                        vector<pair<double, unsigned>> rank(dim);
                        for (unsigned j = 0; j < dim; ++j) {
                            rank[j].first = y[j];
                            rank[j].second = j;
                        }
                        sort(rank.begin(), rank.end());
                        for (unsigned j = 0; j < dim; ++j) {
                            double lb; //= (rank[j-1].first + rank[j].first) / 2;
                            double ub; // = (rank[j+1].first + rank[j].first) / 2;
                            if (j > 0) {
                                double m = (rank[j].first - rank[j-1].first) * (1.0 - m_margin) * 0.5;
                                lb = rank[j].first - m;//(rank[j-1].first + rank[j].first) / 2;

                            }
                            else {
                                lb = std::numeric_limits<double>::lowest();
                            }
                            if (j < dim - 1) {
                                double m = (rank[j+1].first - rank[j].first) * (1.0 - m_margin) * 0.5;
                                ub = rank[j].first) + m;
                            }
                            else {
                                ub = std::numeric_limits<double>::max();
                            }
                            m[2*j] = lb;
                            m[2*j+1] = ub;
                        }
                    }
                }
            }

            ~DataNode () {
                if (mode() == MODE_TRAIN && m_noise_level > 0) {
                    BOOST_VERIFY(m_noise_task.valid());
                    m_noise_task.wait();
                }
            }

            Array<> const &target () const {
                return m_target;
            }

            vector<string> const &rowNames () const {
                return m_rows;
            }

            vector<string> const &colNames () const {
                return m_cols_exp;
            }

            vector<string> const &targetColNames () const {
                return m_cols_target;
            }

            string const &dir () const {
                return m_dir;
            }

            void predict () {
                if (mode() == MODE_PREDICT) {
                   if (m_done) throw StopIterationException();
                    m_done = true;
                }
                else {
                    data().fill(0);
                    delta().fill(0);
                    m_target.fill(0);
                    unsigned r = 0;
                    role::BatchInput::next([this, &r](unsigned i) {
                        Array<>::value_type *x = data().at(r);
                        unsigned j = 0;
                        if (m_do_exp) {
                            Array<>::value_type const *x_exp = m_exp.at(i);
                            for (unsigned l = 0; l < m_cols_exp.size(); ++l) {
                                x[j++] = x_exp[l];
                            }
                        }
                        if (m_do_copy) {
                            Array<>::value_type const *x_copy = m_copy.at(i);
                            for (unsigned l = 0; l < m_cols_copy.size(); ++l) {
                                x[j++] = x_copy[l];
                            }
                        }
                        /*
                        Array<>::value_type const *x_copy = m_copy.at(i);
                        for (unsigned l = 0; l < m_cols_copy.size(); ++l) {
                            x[j++] = x_copy[l];
                        }
                        */
                        Array<>::value_type *y = m_target.at(r);
                        Array<>::value_type const *y_in = m_all_target.at(i);
                        for (unsigned l = 0; l < m_target.size(1); ++l) {
                            y[l] = y_in[l];
                        }

                        Array<>::value_type *m = m_margins.at(r);
                        Array<>::value_type const *m_in = m_all_margins.at(i);
                        for (unsigned l = 0; l < m_target.size(1) * 2; ++l) {
                            m[l] = m_in[l];
                        }
                        ++r;
                    });
                    if (m_noise_level > 0) {
                        BOOST_VERIFY(m_noise_task.valid());
                        m_noise_task.wait();
                        data().add(m_noise);
                        m_noise_task = std::async(std::launch::async, [this](){this->fill_noise();});
                    }
                }

            }

            virtual Array<double> const &labels () const {
                return m_target;
            }

            virtual Array<double> const &margins () const {
                return m_margins;
            }

            virtual void report (ostream &os) const {
                ArrayNode::report(os);
                os << "target:\t" << m_target.l2() << endl;
            }
        };

        class FeatureSelection: public core::ArrayNode {
            DataNode const *m_data;
            vector<vector<unsigned>> m_map; 
            static const unsigned DIM = 3000;
        public:
            FeatureSelection (Model *model, Config const &config)
                : core::ArrayNode(model, config),
                m_data(findInputAndAdd<DataNode>("input", "data"))
            {
                map<string, unsigned> colsIn;
                map<string, unsigned> colsOut;
                {
                    vector<string> const &cc = m_data->targetColNames();
                    for (unsigned i = 0; i < cc.size(); ++i) {
                        colsOut[cc[i]] = i;
                    }
                }
                {
                    vector<string> const &cc = m_data->colNames();
                    for (unsigned i = 0; i < cc.size(); ++i) {
                        colsIn[cc[i]] = i;
                    }
                }
                ifstream is("/home/wdong/src/dream/datasets/GE_train_top");
                m_map.resize(m_data->targetColNames().size());
                for (;;) {
                    string name;
                    is >> name;
                    if (!is) break;
                    auto it = colsOut.find(name);
                    if (it == colsOut.end()) {
                        // consume the line
                        getline(is, name);
                        continue;
                    }
                    vector<unsigned> &v = m_map[it->second];
                    BOOST_VERIFY(v.empty());
                    v.resize(DIM);
                    for (unsigned j = 0; j < DIM; ++j) {
                        is >> name;
                        auto it = colsIn.find(name);
                        BOOST_VERIFY(it != colsIn.end());
                        v[j] = it->second;
                    }
                    BOOST_VERIFY(is);
                }
                for (unsigned i = 0; i < m_map.size(); ++i) {
                    auto &v = m_map[i];
                    if (v.empty()) {
                        BOOST_VERIFY(m_data->targetColNames()[i] == "____");
                        v.resize(DIM);
                        fill(v.begin(), v.end(), 0);
                    }
                    BOOST_VERIFY(v.size() == DIM);
                }
                vector<size_t> sz;
                m_data->data().size(&sz);
                BOOST_VERIFY(sz.size() == 2);
                sz[1] = m_data->targetColNames().size();
                sz.push_back(DIM);
                resize(sz);
            }

            void predict () {
                unsigned samples = data().size(size_t(0));
                for (unsigned i = 0; i < samples; ++i) {
                    Array<>::value_type const *x = m_data->data().at(i);
                    Array<>::value_type *y = data().at(i);
                    for (auto const &v: m_map) {
                        for (unsigned j: v) {
                            y[0] = x[j];
                            ++y;
                        }
                    }
                }
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

            void report (ostream &os) const {
                vector<role::Stat::Stats> cost;
                role::Stat::means(&cost);
                vector<string> const &names = role::Stat::names();
                BOOST_VERIFY(names.size() == cost.size());
                for (unsigned i = 0; i < names.size(); ++i) {
                    os << setw(16) << names[i]; // << ':'; // << cost[i];
                    auto const &c = cost[i];
                    for (unsigned j = 0; j < c.size(); ++j) {
                        os << setw(16) << c[j];
                    }
                    os << endl;
                }
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

        class RankRegression: public Node, public role::Loss {
            DataNode *m_data;
            core::ArrayNode *m_input;
        public:
            RankRegression (Model *model, Config const &config) 
                : Node(model, config),
                  m_data(findInputAndAdd<DataNode>("data", "data")),
                  m_input(findInputAndAdd<core::ArrayNode>("input", "input")),
            {
                role::Loss::init({"loss", "error"});
            }

            void predict () {
                Array<>::value_type const *x = m_input->data().addr();
                Array<>::value_type const *y = m_data->target().addr();
                Array<>::value_type const *m = m_data->margins().addr();
                vector<size_t> sz_x;
                vector<size_t> sz_y;
                vector<size_t> sz_m;
                m_input->data().size(&sz_x);
                m_data->target().size(&sz_y);
                m_data->margins().size(&sz_m);
                BOOST_VERIFY(sz_x.size() == 2);
                BOOST_VERIFY(sz_x == sz_y);
                BOOST_VERIFY(sz_m.size() == 3);
                BOOST_VERIFY(sz_x[0] == sz_m[0]);
                BOOST_VERIFY(sz_x[1] == sz_m[1]);
                BOOST_VERIFY(sz_m[2] == 2);
                for (unsigned row = 0; row < sz_x[0]; ++row) {
                    double l = 0;
                    for (unsigned col = 0; col < sz_x[1]; ++col) {
                        double diff = 0;
                        if (x[0] < m[0]) {
                            diff = y[0] - x[0];
                        }
                        else if (x[0] > m[1]) {
                            diff = x[0] - y[1];
                        }
                        l += 0.5 * diff * diff;
                        acc(1)(diff);
                        ++x;
                        ++y;
                        m += 2;
                    }
                    acc(0)(l);
                }
            }

            void update () {
                Array<>::value_type const *x = m_input->data().addr();
                Array<>::value_type *dx = m_input->delta().addr();
                Array<>::value_type const *y = m_data->target().addr();
                Array<>::value_type const *m = m_data->margins().addr();
                vector<size_t> sz_x;
                m_input->data().size(&sz_x);
                size_t total = sz_x[0] * sz_x[1];
                for (size_t i = 0; i < total; ++i) {
                    if (x[0] < m[0]) {
                        dx[0] += x[0] - y[0];
                    }
                    else if (x[0] > m[1]) {
                        dx[0] += x[0] - y[1];
                    }
                    ++x;
                    ++dx;
                    ++y;
                    m += 2;
                }
            }

            void report (ostream &os) const {
                os << name() << ":\ttarget/" << m_data->target().l2() << "\tpredict/" <<  m_input->data().l2() << endl;
            }
        };
    }
}

#endif

