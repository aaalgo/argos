#include <sstream>
#include <fstream>
#include <stack>
#include <boost/lexical_cast.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include "argos.h"

namespace argos {

    using namespace std;
    using boost::lexical_cast;

    void LoadConfig (string const &path, Config *config) {
        boost::property_tree::read_xml(path, *config);
    }

    static char const *METHOD_NAMES[] = {"NONE", "PREDICT", "PREUPDATE", "UPDATE"};

    ostream &operator << (ostream &os, Plan::TaskId const &task) {
        Node const *node = task.first;
        Method method = task.second;
        os << node->name() << ':' << node->type() << ":" << METHOD_NAMES[method];
        return os;
    }

    Plan::Plan (Model const &model): frozen(false) {
        for (auto node: model.m_nodes) {
            const_cast<Node *>(node)->prepare(this);
        }
        for (unsigned i = 0; i < tasks.size(); ++i) {
            for (TaskId id: tasks[i].inputs) {
                auto it = lookup.find(id);
                BOOST_VERIFY(it != lookup.end());
                tasks[it->second].outputs.push_back(i);
            }
        }
        frozen = true;
    }

    void Plan::print (ostream &os) const {
        for (auto const &task: tasks) {
            os << task.id << endl;
            for (auto const &id: task.inputs) {
                os << '\t' << id << endl;
            }
        }
    }


    void Plan::run (bool dry) const {
        BOOST_VERIFY(frozen);
        unsigned n = tasks.size();
        unsigned done = 0;
        vector<unsigned> n_left(n);
        stack<unsigned> ready;
        
        // topologicall sort
        for (unsigned i = 0; i < n; ++i) {
            n_left[i] = tasks[i].inputs.size();
            if (n_left[i] == 0) {
                ready.push(i);
            }
        }

        while (!ready.empty()) {
            unsigned idx = ready.top();
            ready.pop();
            Task const &task = tasks[idx];
            LOG(debug) << "RUN " << task.id;
            if (!dry) {
                task.callback();
            }
            for (unsigned o: task.outputs) {
                BOOST_VERIFY(n_left[o] > 0);
                --n_left[o];
                if (n_left[o] == 0) {
                    ready.push(o);
                }
            } 
            ++done;
        }
        BOOST_VERIFY(done == n);
    }

    Node::Node (Model *model, Config const &config)
        : m_config(config), m_model(model), m_name(config.get<string>("name", "")), m_type(config.get<string>("type"))
    {
        if (m_name.empty()) {
            m_name = "$"+lexical_cast<string>(this);
        }
        LOG(debug) << "CREATE " << name() << ':' << type();
    }

    Node::~Node () {
    }

    void Node::prepare (Plan *plan) {
        switch (mode()) {
            case MODE_TRAIN:
                {
                    // add preupdate task
                    plan->add(this, TASK_PREUPDATE, std::bind(&Node::preupdate, dynamic_cast<Node *>(this))).add(this, TASK_PREDICT);
                    // add update task
                    auto deps = plan->add(this, TASK_UPDATE, std::bind(&Node::update, dynamic_cast<Node *>(this)));
                    deps.add(this, TASK_PREUPDATE);
                    for (auto &pin: m_outputs) {
                        deps.add(pin.node, TASK_UPDATE);
                    }
                    for (auto &pin: m_inputs) {
                        deps.add(pin.node, TASK_PREUPDATE);
                    }
                }
                // no break, need to add PREDICT tasks
            case MODE_PREDICT:
                {
                    auto deps = plan->add(this, TASK_PREDICT, std::bind(&Node::predict, this));
                    for (auto pin: m_inputs) {
                        deps.add(pin.node, TASK_PREDICT);
                    }
                }
                break;
            default:
                throw runtime_error("mode not supported");
        }
    }

    void Node::load (istream &is) {
    }

    void Node::save (ostream &os) const {
    }

    void Node::init () {
    }

    void Node::sync (Node const *from) {
        stringstream ss;
        from->save(ss);
        ss.seekg(0);
        load(ss);
    }

    void Node::predict () {
    }

    void Node::preupdate () {
    }

    void Node::update () {
    }

    Library library;

    Model::Model (Config const &config, Mode mode): m_config(config), m_mode(mode), m_random(config.get<Random::result_type>("argos.global.seed", 2011)) {
        for (Config::value_type const &node: config.get_child("argos")) {
            string const &name = node.first;
            if (name != "node") continue; // throw runtime_error("xml error");
            Config const &conf = node.second;
            createNode(conf);
        }
        m_input = nullptr;
        m_loss = nullptr;
        for (Node *node: m_nodes) {
            role::Input *input = dynamic_cast<role::Input *>(node);
            if (input) {
                if (m_input) throw runtime_error("can have only one node of role input");
                m_input = input;
            }
            role::Loss *loss = dynamic_cast<role::Loss *>(node);
            if (loss) {
                if (m_loss) throw runtime_error("can have only one node of role loss");
                m_loss = loss;
            }
            role::Stat *stat = dynamic_cast<role::Stat *>(node);
            if (stat) {
                m_stats.push_back(stat);
            }
        }
    }

    Model::~Model () {
        for (Node *node: m_nodes) {
            delete node;
        }
    }

    Node *Model::createNodeHelper (Config const &config) {
        NodeFactory *fac = library.find(config.get<string>("type"));
        Node *node = fac->create(this, config);
        string name = node->name();
        if (!name.empty()) {
            auto r = m_lookup.insert(make_pair(name, node));
            if (r.second) {
                m_nodes.push_back(node);
            }
            else {
                if (r.first->second != node) {
                    throw runtime_error("node '" + name + "' already exists.");
                }
            }
        }
        return node;
    }

    Node *Model::findNodeHelper (string const &name) {
        size_t off = name.find('.');
        if (off == string::npos) {
            auto it = m_lookup.find(name);
            if (it == m_lookup.end()) {
                return nullptr;
            }
            return it->second;
        }
        else {
            auto it = m_lookup.find(name.substr(0, off));    // main name
            if (it == m_lookup.end()) {
                return nullptr;
            }
            Node *node = it->second;
            return node->findInput(name.substr(off + 1)); // pin tag
        }
    }

    void Model::save (string const &path) const {
        ofstream os(path.c_str(), ios::binary);
        for (Node const *node: m_nodes) {
            node->save(os);
        }
    }

    void Model::load (string const &path) {
        ifstream is(path.c_str(), ios::binary);
        for (Node *node: m_nodes) {
            node->load(is);
        }
    }

    void Model::init () { // init model
        for (Node *node: m_nodes) {
            node->init();
        }
    }

    void Model::train (ostream &os) {
        Plan plan(*this);
        unsigned report = config().get<unsigned>("argos.global.report", 0);
        unsigned snapshot = config().get<unsigned>("argos.global.snapshot", 0);
        unsigned maxloop = config().get<unsigned>("argos.global.maxloop", 0);
        string model_path = config().get<string>("argos.global.model", "");
        unsigned loop = 0;
        for (;;) {
            plan.run();
            ++loop;
            if (report && (loop % report == 0)) {
                os << loop << ' ' << endl;
                os << "\ttrain ";
                this->report(os, true);
                // run evaluation
                //clone.sync(*this);
            }
            if (snapshot && (loop % snapshot == 0)) {
                if (model_path.size()) {
                    save(model_path + "." + lexical_cast<string>(loop / snapshot));
                }
            }
            if (maxloop > 0 && loop >= maxloop) break;
        }
    }

    void Model::predict (ostream &os) {
        Plan plan(*this);
        m_input->rewind();
        for (;;) {
            try {
                plan.run();
            }
            catch (StopIterationException) {
                break;
            }
        }
    }

    void Model::report (ostream &os, bool reset) {
        for (role::Stat *stat: m_stats) {
            Node const *node = dynamic_cast<Node const *>(stat);
            BOOST_VERIFY(node);
            vector<double> cost;
            stat->means(&cost);
            vector<string> const &names = stat->names();
            os << node->name();
            BOOST_VERIFY(names.size() == cost.size());
            for (unsigned i = 0; i < names.size(); ++i) {
                os << ' ' << names[i] << ':' << cost[i];
            }
            os << endl;
            if (reset) stat->reset();
        }
    }
}

