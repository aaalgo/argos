#include <argos/argos.h>
#include <argos/neural.h>
#include <argos/combo.h>
#include <argos/io.h>

namespace argos {

    void Model::registerAllFactories () {
        registerFactory("generic", new NodeFactoryImpl<Node>);
        registerFactory("id", new NodeFactoryImpl<neural::FunctionNode<neural::function::id>>);
        registerFactory("relu", new NodeFactoryImpl<neural::FunctionNode<neural::function::relu>>);
        registerFactory("softrelu", new NodeFactoryImpl<neural::FunctionNode<neural::function::softrelu>>);
        registerFactory("tanh", new NodeFactoryImpl<neural::FunctionNode<neural::function::tanh>>);
        registerFactory("logistic", new NodeFactoryImpl<neural::FunctionNode<neural::function::logistic>>);
        registerFactory("param", new NodeFactoryImpl<neural::ParamNode>);
        registerFactory("linear", new NodeFactoryImpl<neural::LinearNode>);
        //registerFactory("input", new NodeFactoryImpl<neural::InputNode>);
        registerFactory("labeltap", new NodeFactoryImpl<neural::LabelTap>);
        //registerFactory("gaussian", new NodeFactoryImpl<neural::GaussianOutputNode>);
        registerFactory("logp", new NodeFactoryImpl<neural::LogPOutputNode>);
        registerFactory("hinge", new NodeFactoryImpl<neural::HingeLossOutputNode>);
        registerFactory("window", new NodeFactoryImpl<neural::WindowNode>);
        registerFactory("max", new NodeFactoryImpl<neural::PoolNode<neural::pool::max>>);
        registerFactory("avg", new NodeFactoryImpl<neural::PoolNode<neural::pool::avg>>);
        registerFactory("pad", new NodeFactoryImpl<neural::PadNode>);
        registerFactory("softmax", new NodeFactoryImpl<neural::SoftMaxNode>);
        registerFactory("dropout", new NodeFactoryImpl<neural::DropOutNode>);
        registerFactory("conv", new neural::ConvNodeFactory);
        registerFactory("global", new neural::GlobalNodeFactory);
        registerFactory("simple", new NodeFactoryImpl<neural::SimpleArrayInputNode>);
        registerFactory("svm", new neural::SvmNodeFactory);
    }
}

