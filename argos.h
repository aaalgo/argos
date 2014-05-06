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
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/log/trivial.hpp>

#define LOG(x) BOOST_LOG_TRIVIAL(x)

namespace argos {

    using namespace std;
    namespace ba = boost::accumulators;

    /// JSON- or XML-like configuration.
    typedef boost::property_tree::ptree Config; 

    /// Load configuration from XML file.
    void LoadConfig (string const &path, Config *);

    /// Running out of data, like Python's counterpart.
    /** 
     *  This is only used in prediction.  For training, the same data
     *  are iterated over again and again and we'll never run out of data.
     *  In prediction mode, when all data are processed, the data-loading
     *  mode will throw this exception, and the code driving the batch
     *  iterations will stop.
     */
    class StopIterationException {
    };

    /// Mode to run the model.
    enum Mode {
        MODE_PREDICT = 1,
        MODE_TRAIN =   2, 
    };

    /// Node method
    /**
     * A node can implement multiple methods, and each method is identified with
     * a number of this type.  The three common methods are predict, pre-update
     * and update.
     */
    enum Method {
        TASK_PREDICT = 1,
        TASK_PREUPDATE = 2,
        TASK_UPDATE = 3,
    };

    class Node;
    class Model;

    /// Network running plan (predict or train).
    /**
     * A plan is a di-graph of tasks, each identified by a TaskID,
     * and has an attached callback function.  An arc "A -> B" means
     * task "B" depends on "A".  A task is executed by having its callback function
     * invoked.  A task can only be executed when all the tasks it depends
     * on have been executed (the task is called to be in a "ready" state). 
     * We define TaskID as pair<Node*, Method>.
     * The implementation of different nodes have to agree upon the meaning
     * of Method, and have to ensure that the same TaskID is always mapped to
     * the same method call on the same Node object.
     *
     * The scheduling of a plan can be done via topological sort of the graph.
     * For now, we only do sequantial scheduling.  But we can potentially 
     * support runing all ready tasks in parallel.
     *
     */
    class Plan {
    public:
        typedef pair<Node const *, Method> TaskId;  // use node ptr and method tag to uniquely identify a method
    private:
        struct Task {
            TaskId id;
            function<void ()> callback;
            // The tasks this one depends on.
            vector<TaskId> inputs;
            // The tasks that depend on this one, identified by index into the 
            // "tasks" vector below.
            vector<unsigned> outputs;
        };
        Mode mode;
        bool frozen;
        vector<Task> tasks;
        // mapping TaskId to index to the "tasks" vector.
        map<TaskId, unsigned> lookup;
    public:
        /// Constructor, from model and mode.
        Plan (Model const &model, Mode m = MODE_PREDICT);

        friend class Deps;
        // A helper class to facilitate adding dependencies.
        class Deps {
            Plan *plan;
            unsigned task;
        public:
            Deps (Plan *p, unsigned t): plan(p), task(t) {
            }
            Deps &add (Node const *node, Method method) {
                plan->tasks[task].inputs.push_back(make_pair(node, method));
                return *this;
            }
        };

        /// Add a task to the plan.
        /**
         * The return value is a Deps object for adding dependencies of this
         * task.  For example,
         *
         *     plan.add(node, METHOD_PREDICT, [](){...})  // adding the task itself
         *    .add(input_node_1, METHOD_PREDICT)          // add dependency 1.
         *    .add(input_node_2, METHOD_PREDICT);         // add dependency 2.
         *
         * A dependency can be added without the depending task being previous added to the 
         * plan.  All reference to tasks to be added in future is resolved after
         * all tasks have been added.
         */
        Deps add (Node const *node, Method method, function<void()> const &callback) {
            BOOST_VERIFY(!frozen);
            unsigned idx = tasks.size();
            Task task;
            task.id = make_pair(node, method);
            task.callback = callback;
            BOOST_VERIFY(lookup.find(task.id) == lookup.end());
            lookup[task.id] = tasks.size();
            tasks.push_back(task);
            return Deps(this, idx);
        }

        /// Print plan to the screen.
        void print (ostream &) const;
        /// Run the plan.
        void run (bool dry = false) const;
    };


    /// Node factory.
    /** The system keeps a registry that maps strings to node factories,
     * so nodes can be dynamically created from a config file.
     */
    struct NodeFactory {
        virtual ~NodeFactory() {}
        /// Create a node from configuration.
        virtual Node *create (Model *model, Config const &config) const = 0;
        /// Test if the given pointer is a instance of class of node created by this factory.
        // using dynamic_cast.
        virtual bool isa (Node const*) const = 0;
    };

    /// Template class to implement a factory for a give node class.
    /**
     * Used like this
     *      NodeFactory *fac = new NodeFactoryImpl<neural::LogPOutputNode>;
     */
    template <typename NODE_TYPE>
    struct NodeFactoryImpl: public NodeFactory {
    public:
        virtual Node *create (Model *model, Config const &config) const {
            return new NODE_TYPE(model, config);
        }
        virtual bool isa (Node const *node) const {
            return dynamic_cast<NODE_TYPE const*>(node) != 0;
        }
    };


    /// The abstract node class.
    class Node {
    public:
        /// Input/output pins.
        /** For now only input pins have tags. */
        struct Pin {
            string tag;
            Node *node;
        };
    private:
        Config m_config;
        Model *m_model;     // the model this node belongs to, weak reference.
        string m_name;
        string m_type;
        vector<Pin> m_inputs;
        vector<Pin> m_outputs;
        map<string, Node *> m_lookup;   // input pin lookup, output pins do not have tags, cannot be looked up.

    protected:
        ///
        template <typename TYPE>
        TYPE *findInputAndAdd (string const &configAttr, string const &tag);

        void addInput (Node *node, string const &tag) {
            m_inputs.push_back({tag, node});
            if (!tag.empty()) {
                auto r = m_lookup.insert(make_pair(tag, node));
                BOOST_VERIFY(r.second);
            }
            node->m_outputs.push_back({"", this});
        }

    public:
        /// Construct from model and config.
        /** All inherited classes must call this constructor. */
        Node (Model *model, Config const &config);
        virtual ~Node ();

        Model *model () { return m_model; }
        string const &name () const { return m_name; }
        string const &type () const { return m_type; }
        vector<Pin> const &inputs () { return m_inputs; }
        vector<Pin> const &outputs () { return m_outputs; }

        Node *findInput (string const &name) {
            auto it = m_lookup.find(name);
            if (it == m_lookup.end()) {
                return nullptr;
            }
            return it->second;
        }

        /// Prepare the node for running, add tasks to plan.
        virtual void prepare (Mode, Plan *);
        /// save parameters/status to stream  
        virtual void save (ostream &os) const;
        /// load parameters/status from stream
        /** Before running tasks, either load or init must be called. */
        virtual void load (istream &is);
        /// initialize parameters/status.
        /** Before running tasks, either load or init must be called. */
        virtual void init ();
        /// Task method.
        virtual void predict (Mode mode);
        /// Task method.
        virtual void preupdate (Mode mode);
        /// Task method.
        virtual void update (Mode mode);
    };

    /// Within the namespace role are several interfaces that a node can implement.
    /**
     * The model with test if a node implement certain role, and invoke the
     * correponding method of that role when appropriate.  For example,
     * when a new batch starts, the model will invoke "reset" of all Input
     * nodes.
     *
     * Every interface in this category have to inherit from the base class
     * Role.
     */
    namespace role {

        class Role {
        public:
            virtual ~Role () {}
        };

        // Sample input.
        class Input: public Role {
        public:
            virtual void rewind (Mode mode) = 0;
        };

        /// Label input.
        class LabelInput: public Role {
        public:
            // for training, the return size should always be the same;
            // for prediction, the number of actual available example is returned.
            virtual vector<int> const &labels () const = 0;
        };

        /// Statistics.
        class Stat: public Role {
        protected:
            typedef ba::accumulator_set<float, ba::stats<ba::tag::mean/*, tag::moment<2>*/>> Acc;
            vector<string> m_names;
            vector<Acc> m_accs;
        public:
            Stat (vector<string> const &names): m_names(names), m_accs(names.size()) {
            }
            void reset () {
                for (auto &acc: m_accs) {
                    acc = Acc();
                }
            }
            vector<string> const &names () const {
                return m_names;
            }
            /// Get the means of statistics.
            void means (vector<float> *v) const {
                v->clear();
                for (auto const &acc: m_accs) {
                    v->push_back(ba::mean(acc));
                }
            }
            Acc &acc (unsigned i) {
                return m_accs[i];
            }
        };

        /// Loss interface.
        /** A model can have multiple stat nodes, but can only have one loss
         * node.  The mean of the first stat value is automatically used as loss.
         */
        class Loss: public Stat {
        public:
            Loss (vector<string> const &names): Stat(names) {
            }
            /// Return the loss value.
            float loss () const {
                BOOST_VERIFY(m_accs.size());
                return boost::accumulators::mean(m_accs[0]);
            }
        };


        /// Parameter interface.
        class Params: public Role {
        public:
        };
    }

    /// Model.
    class Model {
    public:
        typedef std::mt19937 Random;
    private:
        /// Factory registry.
        map<string, NodeFactory *> m_fac;
        void registerFactory (string const &name, NodeFactory *fac) {
            BOOST_VERIFY(fac);
            auto r = m_fac.insert(make_pair(name, fac));
            BOOST_VERIFY(r.second);
        }
        void cleanupFactories () {
            for (auto &v: m_fac) {
                delete v.second;
            }
        }
        void registerAllFactories ();
        Config m_config;
    protected:
        vector<Node *> m_nodes;
        role::Input *m_input;
        role::Loss *m_loss;
        vector<role::Stat *> m_stats;
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

        void predict (ostream &os = cerr);
        void report (ostream &os= cerr) const;
        void verify (bool dry = false);

        friend class Plan;
    };

    template <typename TYPE>
    TYPE *Node::findInputAndAdd (string const &configAttr, string const &tag) {
        string node_name = m_config.get<string>(configAttr);
        TYPE *node = m_model->findNode<TYPE>(node_name);
        if (node == nullptr) throw 0;
        addInput(node, tag);
        return node;
    }

    class LabelOutputNode: public Node { //, public AccOutput {
    protected:
        vector<int> m_labels;           // the labels we predict
        role::LabelInput const *m_label_input; 
    public:
        LabelOutputNode (Model *model, Config const &config): Node(model, config)
        {
            m_label_input = model->findNode<role::LabelInput>(config.get<string>("labels"));
            BOOST_VERIFY(m_label_input);
        }
        vector<int> const &labels () const { return m_labels; }
    };

}

#endif

