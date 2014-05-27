#ifndef ARGOS_NODE_COMBO
#define ARGOS_NODE_COMBO

namespace argos {
    namespace combo {
        using namespace argos::core;

        struct ConvNodeFactory: public NodeFactory {
        public:
            virtual Node *create (Model *model, Config const &config) const {
                string name = config.get<string>("name");
                {
                    Config cfg;
                    cfg.put("type", "pad");
                    cfg.put("input", config.get<string>("input"));
                    cfg.put("name", name + "_pad");
                    cfg.put("width", config.get<size_t>("pad"));
                    cfg.put("height", config.get<size_t>("pad"));
                    PadNode *pad = model->createNode<PadNode>(cfg);
                    BOOST_VERIFY(pad);
                }
                {
                    Config cfg;
                    cfg.put("type", "window");
                    cfg.put("input", name + "_pad");
                    cfg.put("name", name + "_window1");
                    cfg.put("bin", config.get<size_t>("bin"));
                    cfg.put("step", config.get<size_t>("step"));
                    WindowNode *window1 = model->createNode<WindowNode>(cfg);
                    BOOST_VERIFY(window1);
                }
                {
                    Config cfg;
                    cfg.put("type", "linear");
                    cfg.put("input", name + "_window1");
                    cfg.put("name", name + "_linear");
                    cfg.put("local", 1);
                    cfg.put("channel", config.get<size_t>("channel"));
                    LinearNode *linear = model->createNode<LinearNode>(cfg);
                    BOOST_VERIFY(linear);
                }
                {
                    Config cfg;
                    cfg.put("type", config.get<string>("neuron"));
                    cfg.put("name", name + "_neuron");
                    cfg.put("input", name + "_linear");
                    ArrayNode *neuron = model->createNode<ArrayNode>(cfg);
                    BOOST_VERIFY(neuron);
                }
                {
                    Config cfg;
                    cfg.put("type", "window");
                    cfg.put("input", name + "_neuron");
                    cfg.put("name", name + "_window2");
                    cfg.put("bin", config.get<size_t>("pool.bin"));
                    cfg.put("step", config.get<size_t>("pool.step"));
                    WindowNode *window2 = model->createNode<WindowNode>(cfg);
                    BOOST_VERIFY(window2);
                }
                {
                    Config cfg;
                    cfg.put("type", config.get<string>("pool.type"));
                    cfg.put("name", name + "_pool");
                    cfg.put("input", name + "_window2");
                    cfg.put("channel", config.get<size_t>("channel"));
                    ArrayNode *pool = model->createNode<ArrayNode>(cfg);
                    BOOST_VERIFY(pool);
                }
                {
                    Config cfg;
                    cfg.put("type", "norm");
                    cfg.put("name", name);
                    cfg.put("input", name + "_pool");
                    ArrayNode *norm = model->createNode<ArrayNode>(cfg);
                    BOOST_VERIFY(norm);
                    return norm;
                }
            }
            virtual bool isa (Node const *node) const {
                return false ; //dynamic_cast<NODE_TYPE const*>(node) != 0;
            }
        };

        class GlobalNodeFactory: public NodeFactory {
        public:
            virtual Node *create (Model *model, Config const &config) const {
                string name = config.get<string>("name");
                {
                    Config cfg;
                    cfg.put("type", "linear");
                    cfg.put("input", config.get<string>("input"));
                    cfg.put("channel", config.get<size_t>("channel"));
                    cfg.put("name", name + "_linear");
                    ArrayNode *linear = model->createNode<LinearNode>(cfg);
                    BOOST_VERIFY(linear);
                }
                {
                    Config cfg;
                    cfg.put("type", config.get<string>("neuron"));
                    cfg.put("input", name + "_linear");
                    cfg.put("name", name);
                    ArrayNode *neuron = model->createNode<ArrayNode>(cfg);
                    BOOST_VERIFY(neuron);
                    return neuron;
                }
            }
            virtual bool isa (Node const *node) const {
                return false ; //dynamic_cast<NODE_TYPE const*>(node) != 0;
            }
        };

        class SvmNodeFactory: public NodeFactory {
        public:
            virtual Node *create (Model *model, Config const &config) const {
                string name = config.get<string>("name");
                {
                    Config cfg;
                    cfg.put("type", "input-libsvm");
                    cfg.put("batch", config.get<string>("batch"));
                    cfg.put("dim", config.get<string>("dim"));
                    cfg.put("train", config.get<string>("train"));
                    cfg.put("test", config.get<string>("test"));
                    cfg.put("name", name + "_input");
                    Node *input = model->createNode<Node>(cfg);
                    BOOST_VERIFY(input);
                }
                {
                    Config cfg;
                    cfg.put("type", "linear");
                    cfg.put("input", name + "_input");
                    cfg.put("name", name + "_linear");
                    cfg.put("channel", config.get<size_t>("channel"));
                    Node *linear = model->createNode<Node>(cfg);
                    BOOST_VERIFY(linear);
                }
                {
                    Config cfg;
                    cfg.put("type", "hinge");
                    cfg.put("input", name + "_linear");
                    cfg.put("labels", name + "_input");
                    cfg.put("name", name);
                    Node *loss = model->createNode<Node>(cfg);
                    BOOST_VERIFY(loss);
                    return loss;
                }
            }
            virtual bool isa (Node const *node) const {
                return false ; //dynamic_cast<NODE_TYPE const*>(node) != 0;
            }
        };
    }
}
#endif
