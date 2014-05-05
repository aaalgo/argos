#ifndef ARGOS_COMBO
#define ARGOS_COMBO

namespace argos {
    namespace neural {
        struct ConvNodeFactory: public Node::Factory {
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
                    cfg.put("name", name);
                    cfg.put("input", name + "_window2");
                    cfg.put("channel", config.get<size_t>("channel"));
                    ArrayNode *pool = model->createNode<ArrayNode>(cfg);
                    BOOST_VERIFY(pool);
                    return pool;
                }
            }
            virtual bool isa (Node const *node) const {
                return false ; //dynamic_cast<NODE_TYPE const*>(node) != 0;
            }
        };

        class GlobalNodeFactory: public Node::Factory {
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
    }
}
#endif
