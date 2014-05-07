#include "argos.h"
#include "node-core.h"
#include "node-combo.h"
#include "node-utils.h"
#include "node-image.h"
#include "node-cifar.h"

namespace argos {

    void Library::registerInternalFactories () {
        registerClass<Node>("generic");
        registerClass<core::Meta>("meta");
        registerClass<core::FunctionNode<core::function::id>>("id");
        registerClass<core::FunctionNode<core::function::relu>>("relu");
        registerClass<core::FunctionNode<core::function::softrelu>>("softrelu");
        registerClass<core::FunctionNode<core::function::tanh>>("tanh");
        registerClass<core::FunctionNode<core::function::logistic>>("logistic");
        registerClass<core::ParamNode>("param");
        registerClass<core::LinearNode>("linear");
        //registerClass<core::InputNode>("input");
        //registerClass<core::GaussianOutputNode>("gaussian");
        registerClass<core::LogPOutputNode>("logp");
        registerClass<core::HingeLossOutputNode>("hinge");
        registerClass<core::WindowNode>("window");
        registerClass<core::PoolNode<core::pool::max>>("max");
        registerClass<core::PoolNode<core::pool::avg>>("avg");
        registerClass<core::PadNode>("pad");
        registerClass<core::SoftMaxNode>("softmax");
        registerClass<core::DropOutNode>("dropout");
        registerClass<utils::LabelTap>("labeltap");
        registerClass<utils::LibSvmInputNode>("input-libsvm");
        registerClass<utils::Eval>("eval");
        registerFactory("conv", new combo::ConvNodeFactory);
        registerFactory("global", new combo::GlobalNodeFactory);
        registerFactory("svm", new combo::SvmNodeFactory);

        registerClass<cifar::CifarInputNode>("input-cifar");
        registerClass<image::ImageInputNode>("input-image");
    }
}

