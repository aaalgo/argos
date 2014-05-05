#include <argos/argos.h>
#include <argos/neural.h>
#include <argos/combo.h>
#include <argos/io.h>

namespace argos {

    template <typename NODE_TYPE>
    struct NodeFactoryImpl: public Node::Factory {
    public:
        virtual Node *create (Model *model, Config const &config) const {
            return new NODE_TYPE(model, config);
        }
        virtual bool isa (Node const *node) const {
            return dynamic_cast<NODE_TYPE const*>(node) != 0;
        }
    };

    void Model::register_factories () {
        m_fac["generic"] = new NodeFactoryImpl<Node>;
        m_fac["id"] = new NodeFactoryImpl<neural::FunctionNode<neural::function::id>>;
        m_fac["relu"] = new NodeFactoryImpl<neural::FunctionNode<neural::function::relu>>;
        m_fac["softrelu"] = new NodeFactoryImpl<neural::FunctionNode<neural::function::softrelu>>;
        m_fac["tanh"] = new NodeFactoryImpl<neural::FunctionNode<neural::function::tanh>>;
        m_fac["logistic"] = new NodeFactoryImpl<neural::FunctionNode<neural::function::logistic>>;
        m_fac["param"] = new NodeFactoryImpl<neural::ParamNode>;
        m_fac["linear"] = new NodeFactoryImpl<neural::LinearNode>;
        //m_fac["input"] = new NodeFactoryImpl<neural::InputNode>;
        m_fac["labeltap"] = new NodeFactoryImpl<neural::LabelTap>;
        //m_fac["gaussian"] = new NodeFactoryImpl<neural::GaussianOutputNode>;
        m_fac["logp"] = new NodeFactoryImpl<neural::LogPOutputNode>;
        m_fac["hinge"] = new NodeFactoryImpl<neural::HingeLossOutputNode>;
        m_fac["window"] = new NodeFactoryImpl<neural::WindowNode>;
        m_fac["max"] = new NodeFactoryImpl<neural::PoolNode<neural::pool::max>>;
        m_fac["avg"] = new NodeFactoryImpl<neural::PoolNode<neural::pool::avg>>;
        m_fac["pad"] = new NodeFactoryImpl<neural::PadNode>;
        m_fac["softmax"] = new NodeFactoryImpl<neural::SoftMaxNode>;
        m_fac["dropout"] = new NodeFactoryImpl<neural::DropOutNode>;
        m_fac["conv"] = new neural::ConvNodeFactory;
        m_fac["global"] = new neural::GlobalNodeFactory;
        m_fac["simple"] = new NodeFactoryImpl<neural::SimpleArrayInputNode>;
    }
}
