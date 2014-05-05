#include <fstream>
#include <stack>
#include <boost/lexical_cast.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include "argos.h"

namespace argos {

    using namespace std;
    using boost::lexical_cast;

    static char const *METHOD_NAMES[] = {"NONE", "INIT", "PREDICT", "PREUPDATE", "UPDATE"};

    void Plan::freeze () {
        for (unsigned i = 0; i < tasks.size(); ++i) {
            for (TaskId id: tasks[i].inputs) {
                auto it = lookup.find(id);
                BOOST_VERIFY(it != lookup.end());
                tasks[it->second].outputs.push_back(i);
            }
        }
    }

    void Plan::display () const {
        for (auto const &task: tasks) {
            Node const *node = task.id.first;
            Method method = task.id.second;
            cout << node->name()  << ':' << node->type() << ' '  << METHOD_NAMES[method] << endl;
            for (auto const &id: task.inputs) {
                Node const *node = id.first;
                Method method = id.second;
                cout << '\t' << node->name()  << ':' << node->type() << ' '  << METHOD_NAMES[method] << endl;
            }
        }
    }


    void Plan::run (bool dry) const {
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
            if (dry) {
                Node const *node = task.id.first;
                Method method = task.id.second;
                cerr << "RUN " << node->name() << ':' << node->type() << "->" << METHOD_NAMES[method] << endl;
            }
            else {
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
        //cerr << "creating " << name() << ':' << type() << endl;
    }

    Node::~Node () {
    }

    void Node::prepare (Mode mode, Plan *plan) {
        switch (mode) {
            case MODE_TRAIN:
            case MODE_VERIFY:
                {
                    // add preupdate task
                    plan->add(this, TASK_PREUPDATE, [this, mode](){this->preupdate(mode);}).add(this, TASK_PREDICT);
                    // add update task
                    auto deps = plan->add(this, TASK_UPDATE, [this, mode](){this->update(mode);});
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
                    auto deps = plan->add(this, TASK_PREDICT, std::bind(&Node::predict, this, mode));//[this, mode](){this->predict(mode);});
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

    void Node::predict (Mode mode) {
    }

    void Node::preupdate (Mode mode) {
    }

    void Node::update (Mode mode) {
    }

    void Model::prepare (Mode mode, Plan *plan) {
        plan->init(mode);
        for (auto node: m_nodes) {
            node->prepare(mode, plan);
        }
        plan->freeze();
    }

    Model::Model (Config const &config): m_config(config), m_random(config.get<Random::result_type>("argos.default.seed", 2011)) {
        register_factories();
        for (Config::value_type const &node: config.get_child("argos")) {
            string const &name = node.first;
            if (name != "node") continue; // throw runtime_error("xml error");
            Config const &conf = node.second;
            createNode(conf);
        }
        m_input = nullptr;
        for (Node *node: m_nodes) {
            Input *input = dynamic_cast<Input *>(node);
            if (input) {
                if (m_input) throw runtime_error("can have only one input");
                m_input = input;
            }
            Stat *stat = dynamic_cast<Stat *>(node);
            if (stat) {
                m_stats.push_back(stat);
            }
        }
    }

    Model::~Model () {
        for (Node *node: m_nodes) {
            delete node;
        }
        cleanup_factories();
    }

    Node *Model::createNodeHelper (Config const &config) {
        Node::Factory *fac = nullptr;
        {
            string type = config.get<string>("type");
            auto it = m_fac.find(type);
            if (it == m_fac.end()) {
                throw runtime_error("node type '" + type + "' not found");
            }
            fac = it->second;
        }
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

    /*
    void Model::train () {
        Plan plan;
        schedule(MODE_TRAIN, &plan);
        for (int i = 0; i < loops; ++i) {
            load_data();
            run(plan);
        }
        Plan plan;
        model.prepare(MODE_TRAIN, &plan);
        unsigned loop = 0;
        for (;;) {
            plan.run();
            ++loop;
            if (report && (loop % report == 0)) {
                cerr << loop << ' ';
                model.report(cerr);
            }
            if (snapshot && (loop % snapshot == 0)) {
                model.save(model_path + "." + lexical_cast<string>(loop / snapshot));
            }
        }
    }
    */

    /*
    void Model::verify (bool dry) {
        Plan plan;
        prepare(MODE_VERIFY, &plan);
        plan.run(dry);
    }
    */

    void Model::predict (ostream &os) {
        Plan plan;
        prepare(MODE_PREDICT, &plan);
        m_input->reset(MODE_PREDICT);
        for (auto *stat: m_stats) {
            stat->reset();
        }
        for (;;) {
            try {
                plan.run();
            }
            catch (StopIterationException) {
                break;
            }
        }
        report(os);
        for (auto *stat: m_stats) {
            stat->reset();
        }
    }

    void Model::report (ostream &os) const {
        for (Stat const *stat: m_stats) {
            Node const *node = dynamic_cast<Node const *>(stat);
            BOOST_VERIFY(node);
            vector<float> cost;
            stat->cost(&cost);
            vector<string> const &names = stat->names();
            os << node->name();
            BOOST_VERIFY(names.size() == cost.size());
            for (unsigned i = 0; i < names.size(); ++i) {
                os << ' ' << names[i] << ':' << cost[i];
            }
            os << endl;
        }
    }

    void load_config (string const &path, Config *config) {
        boost::property_tree::read_xml(path, *config);
    }
}

