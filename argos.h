#ifndef WDONG_ARGOS
#define WDONG_ARGOS

#include <unistd.h>
#include <array>
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
#include <boost/accumulators/statistics/moment.hpp>
#include <boost/accumulators/statistics/count.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/log/trivial.hpp>
#define LOG(x) BOOST_LOG_TRIVIAL(x)

#include <http++.h>

namespace argos {

    //bool isatty = ::isatty(1);

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
        TASK_USER = 4
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
        bool frozen;
        vector<Task> tasks;
        // mapping TaskId to index to the "tasks" vector.
        map<TaskId, unsigned> lookup;
    public:
        /// Constructor, from model and mode.
        Plan (Model const &model);

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
        Deps add (Node const *node, Method method, function<void()> callback) {
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
    /** Node must be constructed with a fixed mode -- the mode of the model. */
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
        /// Retrieve name from config, and add node with that name as input.
        template <typename TYPE>
        TYPE *findInputAndAdd (string const &configAttr, string const &tag, string const &defaul = "");

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

        /// get config with fallback
        /** First try to get path from config of this node.
         * If fails, try to get fallback from model config.
         */
        template <typename TYPE>
        TYPE getConfig (string const &path, string const &fallback) const;
        template <typename TYPE>
        TYPE getConfig (string const &path, string const &fallback, TYPE defaul) const;

        Model *model () { return m_model; }
        Mode mode () const;
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
        virtual void prepare (Plan *);
        /// save parameters/status to stream  
        virtual void save (ostream &os) const;
        /// load parameters/status from stream
        /** Before running tasks, either load or init must be called. */
        virtual void load (istream &is);
        /// initialize parameters/status.
        /** Before running tasks, either load or init must be called. */
        virtual void init ();
        /// copy parameters/status from given node.
        /** The given node must be of the same type and specification.
         * Equivalent to the following:
         *   stringstream ss;
         *   from->save(ss);
         *   ss.seekg(0);
         *   load(ss);
         */ 
        virtual void sync (Node const *from);
        /// Task method.
        virtual void predict ();
        /// Task method.
        virtual void preupdate ();
        /// Task method.
        virtual void update ();

        virtual void report (ostream &os) const {
        }

        virtual void handle (http::server::request const &req, http::server::reply &rep) const {
        }
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
     *
     * Virtual inheritance is used to solve the diamond problem, all classes
     * under the role namespace are not allowed to have a constructor.  Rather,
     * a "init" method is used when needed.
     */
    namespace role {

        class Role {
        public:
            virtual ~Role () {}
        };

        // Sample input.
        class Input: public virtual Role {
        public:
            virtual void rewind () = 0;
        };

        class BatchInput: public virtual Input {
            unsigned m_mode;
            unsigned m_batch;
            unsigned m_off;
            vector<unsigned> m_index;
        public:
            void init (unsigned batch, unsigned sz, Mode mode) {
                m_mode = mode;
                m_batch = batch;
                m_off = 0;
                m_index.resize(sz);
                for (unsigned i = 0; i < sz; ++i) m_index[i] = i;
                if (m_mode  == MODE_TRAIN) {
                    m_off = sz; // so that index will be shuffled when next is called
                }
            }
            unsigned batch () const {
                return m_batch;
            }
            void rewind () {
                m_off = 0;
            }
            void next (function<void(unsigned)> callback) {
                if (m_off + m_batch > m_index.size()) {
                    if (m_mode == MODE_TRAIN) {
                        rewind();
                        random_shuffle(m_index.begin(), m_index.end());
                        BOOST_VERIFY(m_off + m_batch <= m_index.size());
                    }
                }
                if (m_off >= m_index.size()) {
                    throw StopIterationException();
                }
                for (unsigned b = 0; b < m_batch; ++b) {
                    if (m_off >= m_index.size()) break;
                    callback(m_index[m_off++]);
                }
            }
        };

        /// Label input.
        template <typename T=int>
        class LabelInput: public virtual Role {
        public:
            // for training, the return size should always be the same;
            // for prediction, the number of actual available example is returned.
            virtual vector<T> const &labels () const = 0;
        };

        /// Statistics.
        class Stat: public virtual Role {
        public:
            typedef array<double, 6> Stats;
        protected:
            typedef ba::accumulator_set<double, ba::stats<ba::tag::mean, ba::tag::min, ba::tag::max, ba::tag::count, ba::tag::variance, ba::tag::moment<2>>> Acc;
            vector<string> m_names;
            vector<Acc> m_accs;

            Stats stats (Acc const &acc) const {
                Stats st;
                st[0] = ba::mean(acc);
                st[1] = ba::moment<2>(acc);
                st[2] = sqrt(ba::variance(acc));
                st[3] = ba::min(acc);
                st[4] = ba::max(acc);
                st[5] = ba::count(acc);
                return st;
            }
        public:
            void init (vector<string> const &names) {
                m_names = names;
                m_accs.resize(names.size());
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
            void means (vector<Stats> *v) const {
                v->clear();
                for (auto const &acc: m_accs) {
                    v->push_back(stats(acc));
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
        class Loss: public virtual Stat {
        public:
            /// Return the loss value.
            double loss () const {
                BOOST_VERIFY(m_accs.size());
                return ba::mean(m_accs[0]);
            }
        };

        /// Parameter interface for gradient checking.
        /**
         * Gradient checking algroithm:
         *  1. do a normal back-propagation pass to obtain the normal deltas.
         *  2. For each parameter
         *      * do a forward pass, record loss
         *      * perturb the parameter
         *      * do another forward pass, record loss
         *      * compute gradient, check with delta
         *      * rollback to old value (un-perturb)
         */
        class Params: public virtual Role {
        public:
            /// size of parameters.
            virtual size_t dim () const = 0;
            /// perturbe parameter with epsilon.
            /** returns old value to be used for rollback.*/
            virtual void perturb (size_t index, double epsilon) = 0;
            virtual double gradient (size_t index) const = 0;
            virtual double value (size_t index) const = 0;
        };
    }

    /// Node factory library.
    class Library {
        /// Record all modules we have opened.
        map<string, void *> m_modules;
        /// Factory registry.
        map<string, NodeFactory *> m_fac;
        void cleanupFactories () {
            for (auto &v: m_fac) {
                delete v.second;
            }
        }
        void registerInternalFactories ();
        void cleanupModules ();
    public:
        Library () {
            registerInternalFactories();
        }
        ~Library () {
            cleanupFactories();
            cleanupModules();
        }
        /// Register a factory.
        void registerFactory (string const &name, NodeFactory *fac) {
            BOOST_VERIFY(fac);
            auto r = m_fac.insert(make_pair(name, fac));
            BOOST_VERIFY(r.second);
        }
        /// Register a default factory implementation for a node class.
        template <typename T>
        void registerClass (string const &name) {
            registerFactory(name, new NodeFactoryImpl<T>);
        }
        /// A shared library must export entry function of this type, named "ArgusRegisterLibrary".
        typedef void (*ArgosRegisterLibraryFunctionType) (Library *);
        /// Load factories from a shared library.
        void load (string const &path);
        /// find a node factory by name.
        NodeFactory *find (string const &type) {
            auto it = m_fac.find(type);
            if (it == m_fac.end()) {
                throw runtime_error("node type '" + type + "' not found");
            }
            return it->second;
        }
    };

    /// The library singleton.
    extern Library library;

    /// Model.
    class Model {
    public:
        typedef std::mt19937 Random;
    private:
        Config m_config;
        Mode m_mode;

    protected:
        vector<Node *> m_nodes;
        role::Input *m_input;
        role::Loss *m_loss;
        vector<role::Stat *> m_stats;
        map<string, Node *> m_lookup;
        Random m_random;

        Node *findNodeHelper (string const &name);
        Node *createNodeHelper (Config const &config);

        bool m_run_server;
        http::server::server *m_server;

        void startServer ();
        void stopServer () {
            m_server->wait_stop();
            delete m_server;
        }
    public:
        Model (Config const &config, Mode mode);
        ~Model ();

        Model (Model const &from, Mode mode): Model(from.config(), mode) {
        }

        Mode mode () const { return m_mode;}

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
        /// Synchronize from another model of the same architecture.
        void sync (Model const &from) {
            BOOST_VERIFY(m_nodes.size() == from.m_nodes.size());
            for (unsigned i = 0; i < m_nodes.size(); ++i) {
                m_nodes[i]->sync(from.m_nodes[i]);
            }
        }

        void predict (ostream &os = cerr);
        void train (ostream &os = cerr);
        /// Report all statistics
        /** If reset, then the statistics are reset to 0 after being reported.
         */
        void report (ostream &os= cerr, bool reset = false);
        /// Gradient verification, node must be of role Params.
        void verify (string const &node, double epsilon, size_t sample = 0);

        friend class Plan;
    };

    template <typename TYPE>
    TYPE *Node::findInputAndAdd (string const &configAttr, string const &tag, string const &defaul) {
        string node_name;
        if (defaul.empty()) {
            node_name = m_config.get<string>(configAttr);
        }
        else {
            node_name = m_config.get<string>(configAttr, defaul);
        }
        TYPE *node = m_model->findNode<TYPE>(node_name);
        if (node == nullptr) {
            LOG(error) << "cannot find node " << node_name; 
            throw 0;
        }
        addInput(node, tag);
        return node;
    }

    template <typename TYPE>
    TYPE Node::getConfig (string const &path, string const &fallback) const {
        try {
            return m_config.get<TYPE>(path);
        }
        catch (...) {
            return m_model->config().get<TYPE>(fallback);
        }
    }

    template <typename TYPE>
    TYPE Node::getConfig (string const &path, string const &fallback, TYPE defaul) const {
        try {
            return m_config.get<TYPE>(path);
        }
        catch (...) {
            return m_model->config().get<TYPE>(fallback, defaul);
        }
    }

    inline Mode Node::mode () const {
        return m_model->mode();
    }

    template <typename T>
    class LabelOutputNode: public Node { //, public AccOutput {
        role::LabelInput<T> const *m_label_input; 
    protected:
        vector<T> m_labels;           // the labels we predict
    public:
        LabelOutputNode (Model *model, Config const &config): Node(model, config)
        {
            m_label_input = model->findNode<role::LabelInput<T>>(config.get<string>("label"));
            BOOST_VERIFY(m_label_input);
        }
        vector<T> const &labels () const { return m_labels; }
        vector<T> const &inputLabels () const { return m_label_input->labels(); }
    };
}

#endif

