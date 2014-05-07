#include <argos/argos.h>
#include <argos/neural.h>
#include <argos/combo.h>
#include <argos/io.h>
#include <argos/eval.h>

namespace argos {

    void Library::registerInternalFactories () {
        registerClass<Node>("generic");
        registerClass<neural::FunctionNode<neural::function::id>>("id");
        registerClass<neural::FunctionNode<neural::function::relu>>("relu");
        registerClass<neural::FunctionNode<neural::function::softrelu>>("softrelu");
        registerClass<neural::FunctionNode<neural::function::tanh>>("tanh");
        registerClass<neural::FunctionNode<neural::function::logistic>>("logistic");
        registerClass<neural::ParamNode>("param");
        registerClass<neural::LinearNode>("linear");
        //registerClass<neural::InputNode>("input");
        registerClass<neural::LabelTap>("labeltap");
        //registerClass<neural::GaussianOutputNode>("gaussian");
        registerClass<neural::LogPOutputNode>("logp");
        registerClass<neural::HingeLossOutputNode>("hinge");
        registerClass<neural::WindowNode>("window");
        registerClass<neural::PoolNode<neural::pool::max>>("max");
        registerClass<neural::PoolNode<neural::pool::avg>>("avg");
        registerClass<neural::PadNode>("pad");
        registerClass<neural::SoftMaxNode>("softmax");
        registerClass<neural::DropOutNode>("dropout");
        registerClass<neural::SimpleArrayInputNode>("simple");
        registerClass<Eval>("eval");
        registerFactory("conv", new neural::ConvNodeFactory);
        registerFactory("global", new neural::GlobalNodeFactory);
        registerFactory("svm", new neural::SvmNodeFactory);
    }
}

