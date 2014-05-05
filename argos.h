#ifndef WDONG_ARGOS
#define WDONG_ARGOS

#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include <map>
#include <random>
#include <functional>
#include <boost/assert.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/log/trivial.hpp>

#define LOG(x) BOOST_LOG_TRIVIAL(x)

namespace argos {

    using namespace std;

    typedef boost::property_tree::ptree Config; 

    void load_config (string const &path, Config *);

    class StopIterationException {
    };

    // Mode to run the model.
    enum Mode {
        MODE_PREDICT = 1,
        MODE_TRAIN =   2, 
        MODE_VERIFY =  3,    // MODE_VERIFY is the same as MODE_TRAIN except:
                             //      it's not to be to run iteratively.
                             //      parameters do not have their values updated.
    };

    enum Method {
        TASK_PREDICT = 1,
        TASK_PREUPDATE = 2,
        TASK_UPDATE = 3,
    };

    class Node;

    // network execution plan
    class Plan {
        Mode mode;
        typedef pair<Node const *, Method> TaskId;  // use node ptr and method tag to uniquely identify a method
        struct Task {
            TaskId id;
            function<void ()> callback;
            vector<TaskId> inputs;
            vector<unsigned> outputs;
        };
        vector<Task> tasks;
        map<TaskId, unsigned> lookup;
    public:
        friend class Deps;
        class Deps {
            Plan *plan;
            unsigned task;
        public:
            Deps (Plan *p, unsigned t): plan(p), task(t) {
            }
            void add (Node const *node, Method method) {
                plan->tasks[task].inputs.push_back(make_pair(node, method));
            }
        };

        Deps add (Node const *node, Method method, function<void()> const &callback) {
            unsigned idx = tasks.size();
            Task task;
            task.id = make_pair(node, method);
            task.callback = callback;
            BOOST_VERIFY(lookup.find(task.id) == lookup.end());
            lookup[task.id] = tasks.size();
            tasks.push_back(task);
            return Deps(this, idx);
        }

        void init (Mode m) {
            mode = m;
            tasks.clear();
            lookup.clear();
        }
        void freeze ();
        void display () const;
        void run (bool dry = false) const;
    };

    class Model;
    /// The abstract node class.
    class Node {
    public:
        struct Factory {
            virtual ~Factory() {}
            virtual Node *create (Model *model, Config const &config) const = 0;
            virtual bool isa (Node const*) const = 0;
        };
        struct Pin {
            string name;
            Node *node;
        };
    private:
        Config m_config;
        Model *m_model;
        string m_name;
        string m_type;
        vector<Pin> m_inputs;
        vector<Pin> m_outputs;
        map<string, Node *> m_lookup;   // input pin lookup, output lookup is not supported for now

    protected:
        template <typename TYPE>
        TYPE *findInputAndAdd (string const &attr, string const &tag);

        void add_input (Node *node, string const &name) {
            m_inputs.push_back({name, node});
            if (!name.empty()) {
                m_lookup[name] = node;
            }
            node->m_outputs.push_back({"", this});
        }

    public:
        Node (Model *model, Config const &config);

        virtual ~Node ();

        Model *model () {
            return m_model;
        }

        string const &name () const {
            return m_name;
        }

        string const &type () const {
            return m_type;
        }

        vector<Pin> &inputs () {
            return m_inputs;
        }

        vector<Pin> &outputs () {
            return m_outputs;
        }

        Node *findInput (string const &name) {
            auto it = m_lookup.find(name);
            if (it == m_lookup.end()) {
                return nullptr;
            }
            return it->second;
        }

        virtual void prepare (Mode, Plan *);
        virtual void load (istream &is);
        virtual void save (ostream &os) const;
        virtual void init ();   // init model
        virtual void predict (Mode mode);
        virtual void preupdate (Mode mode);
        virtual void update (Mode mode);
    };

    class Input {
    public:
        virtual void reset (Mode mode) = 0;
    };

    class Stat {
        vector<string> m_names;
        vector<float> m_values;
        unsigned m_acc;
    protected:
        unsigned &acc () {
            return m_acc;
        }
        vector<float> &scores () {
            return m_values;
        }
        virtual void justToMakeItADynamicClass () const {
        }
    public:
        Stat (vector<string> const &&names): m_names(names), m_values(names.size(), 0), m_acc(0) {
        }
        void reset () {
            fill(m_values.begin(), m_values.end(), 0);
            m_acc = 0;
        }
        vector<string> const &names () const {
            return m_names;
        }
        void cost (vector<float> *v) const {
            v->resize(m_values.size());
            for (unsigned i = 0; i < m_values.size(); ++i) {
                v->at(i) = m_values[i] / m_acc;
            }
        }
        float first_cost () const {
            BOOST_VERIFY(m_values.size());
            return m_values[0] / m_acc;
        }
    };

    class Model {
    public:
        typedef std::mt19937 Random;
    private:
        map<string, Node::Factory *> m_fac;
        void register_factories ();
        void cleanup_factories () {
            for (auto &v: m_fac) {
                delete v.second;
            }
        }

        Config m_config;
    protected:
        vector<Node *> m_nodes;
        Input *m_input;
        vector<Stat *> m_stats;
        map<string, Node *> m_lookup;
        Random m_random;

        Node *findNodeHelper (string const &name);
        Node *createNodeHelper (Config const &config);
    public:
        Model (Config const &config);
        ~Model ();

        Config const &config () const { return m_config; }
        Random &random () { return m_random; }

        template <typename TYPE = Node>
        TYPE *createNode (Config const &config) {
            return dynamic_cast<TYPE *>(createNodeHelper(config));
        }

        template <typename TYPE = Node>
        TYPE *findNode (string const &name) {
            Node *raw_node = findNodeHelper(name);
            if (raw_node == nullptr) return nullptr;
            TYPE *node = dynamic_cast<TYPE *>(raw_node);
            if (node == nullptr) throw runtime_error("node cast error");
            return node;
        }

        void init ();   // initialize
        void load (string const &path); // load
        void save (string const &path) const;
        void prepare (Mode mode, Plan *plan); 
        void verify (bool dry = false);
        void predict (ostream &os = cerr);
        void report (ostream &os= cerr) const;
    };

    template <typename TYPE>
    TYPE *Node::findInputAndAdd (string const &attr, string const &name) {
        string node_name = m_config.get<string>(attr);
        TYPE *node = m_model->findNode<TYPE>(node_name);
        if (node == nullptr) throw 0;
        add_input(node, name);
        return node;
    }

    class LabelInput {
    public:
        // for training, the return size should always be the same;
        // for prediction, the number of actual available example is returned.
        virtual vector<int> const &labels () const = 0;
    };

    class LabelOutputNode: public Node { //, public AccOutput {
    protected:
        vector<int> m_labels;           // the labels we predict
        LabelInput const *m_label_input; 
    public:
        LabelOutputNode (Model *model, Config const &config): Node(model, config)
        {
            m_label_input = model->findNode<LabelInput>(config.get<string>("labels"));
            BOOST_VERIFY(m_label_input);
        }
        vector<int> const &labels () const { return m_labels; }
    };
}

#endif

