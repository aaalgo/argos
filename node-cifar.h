#ifndef ARGOS_NODE_CIFAR
#define ARGOS_NODE_CIFAR

namespace argos {
    namespace cifar {

        struct Example {
            int label;
            int labelx;
            vector<float> data;
        };

        static size_t constexpr WIDTH = 32;
        static size_t constexpr HEIGHT = 32;
        static size_t constexpr AREA = WIDTH * HEIGHT;
        static size_t constexpr CHANNELS = 3;
        static size_t constexpr DIM = WIDTH * HEIGHT * CHANNELS;

        class DataSet: public vector<Example> {
        public:
            void load (string const &path, bool big = false, int cat = 0) {
                ifstream is(path.c_str(), ios::binary);
                size_t sz = big ? 2 : 1 + DIM;
                is.seekg(0, ios::end);
                size_t total = is.tellg();
                is.seekg(0, ios::beg);
                BOOST_VERIFY(total % sz == 0);
                resize(total / sz);
                vector<uint8_t> bytes(sz);
                unsigned n = 0;
                for (unsigned i = 0; i < size(); ++i) {
                    Example &e = at(n);
                    uint8_t *p = &bytes[0];
                    is.read((char *)p, sz);
                    e.label = int(p[0]); ++p;
                    e.labelx = 0;
                    if (big) {
                        e.labelx = int(p[0]);
                        ++p;
                    }
                    e.data.resize(DIM);
                    for (unsigned j = 0; j < CHANNELS; ++j) {
                        for (unsigned k = 0; k < AREA; ++k) {
                            e.data[k * CHANNELS + j] =
                                float(p[j * AREA + k]) * 2/ 255.0 - 1;
                        }
                    }
                    if ((cat == 0) || (e.label < cat)) {
                        ++n;
                    }
                }
                resize(n);
                BOOST_VERIFY(is);
            }
        };

        class CifarInputNode: public core::ArrayNode, public role::LabelInput<int>, public role::BatchInput {
            DataSet m_examples;
            vector<int> m_labels;
        public:
            CifarInputNode (Model *model, Config const &config) 
                : ArrayNode(model, config)
            {
                unsigned batch = getConfig<unsigned>("batch", "argos.global.batch");
                vector<size_t> size{batch, HEIGHT, WIDTH, CHANNELS};
                core::ArrayNode::resize(size);
                core::ArrayNode::setType(core::ArrayNode::IMAGE);

                string path;
                if (mode() == MODE_PREDICT) {
                    path = config.get<string>("test");
                }
                else {
                    path = config.get<string>("train");
                }
                LOG(info) << "loading image paths from " << path;
                m_examples.load(path);
                role::BatchInput::init(batch, m_examples.size(), mode());
            }

            void predict () {
                data().fill(0.0);
                m_labels.clear(); //resize(m_batch);
                Array<>::value_type *x = data().addr();
                role::BatchInput::next([this, &x](unsigned i) {
                    cifar::Example const &e = m_examples[i];
                    m_labels.push_back(e.label);
                    copy(e.data.begin(), e.data.end(), x);
                    x = data().walk<0>(x);
                });
            }

            virtual vector<int> const &labels () const {
                return m_labels;
            }
        };
    }
}

#endif

