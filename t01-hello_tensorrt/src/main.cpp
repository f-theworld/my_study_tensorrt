
// tensorRT include
#include <NvInfer.h>
#include <NvInferRuntime.h>

// cuda include
#include <cuda_runtime.h>

// system include
#include <stdio.h>

class TRTLogger : public nvinfer1::ILogger {
public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override {
        // TensorRT 10 的日志等级建议过滤掉过多的 VERBOSE 信息
        if (severity <= Severity::kINFO) {
            printf("%d: %s\n", (int)severity, msg);
        }
    }
};

nvinfer1::Weights make_weights(float* ptr, int n){
    nvinfer1::Weights w;
    w.count = n;
    w.type = nvinfer1::DataType::kFLOAT;
    w.values = ptr;
    return w;
}

int main(){
    // 本代码主要实现一个最简单的神经网络 figure/simple_fully_connected_net.png 
     
    TRTLogger logger; // logger是必要的，用来捕捉warning和info等
// 注意：10.x 推荐使用 std::unique_ptr 配合自定义删除器，这里为了直观先用普通指针
    auto builder = nvinfer1::createInferBuilder(logger);
    auto config = builder->createBuilderConfig();
    
    // 显式 Batch 模式在 10.x 是强制性的，参数必须为 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)
    // 或者简单使用 1U 表示显式 Batch
    auto network = builder->createNetworkV2(1U);
    // ----------------------------- 1. 定义 builder, config 和network -----------------------------
    // 这是基本需要的组件
    //形象的理解是你需要一个builder去build这个网络，网络自身有结构，这个结构可以有不同的配置

    // 创建一个构建配置，指定TensorRT应该如何优化模型，tensorRT生成的模型只能在特定配置下运行

    // 创建网络定义，其中createNetworkV2(1)表示采用显性batch size，新版tensorRT(>=7.0)时，不建议采用0非显性batch size
    // 因此贯穿以后，请都采用createNetworkV2(1)而非createNetworkV2(0)或者createNetwork


    // 构建一个模型
    /*
        Network definition:

        image
          |
        linear (fully connected)  input = 3, output = 2, bias = True     w=[[1.0, 2.0, 0.5], [0.1, 0.2, 0.5]], b=[0.3, 0.8]
          |
        sigmoid
          |
        prob
    */

    // ----------------------------- 2. 输入，模型结构和输出的基本信息 -----------------------------
    const int num_input = 3;   // in_channel
    const int num_output = 2;  // out_channel
    float layer1_weight_values[] = {1.0, 2.0, 0.5, 0.1, 0.2, 0.5}; // 前3个给w1的rgb，后3个给w2的rgb 
    float layer1_bias_values[]   = {0.3, 0.8};

    //输入指定数据的名称、数据类型和完整维度，将输入层添加到网络
// 输入层：N, C, H, W
    auto input = network->addInput("image", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4(1, num_input, 1, 1));
    nvinfer1::Weights layer1_weight = make_weights(layer1_weight_values, 6);
    nvinfer1::Weights layer1_bias   = make_weights(layer1_bias_values, 2);
    //添加全连接层
    auto layer1 = network->addConvolutionNd(*input, num_output, nvinfer1::Dims2(1, 1), layer1_weight, layer1_bias);
     //添加激活层 
    auto prob = network->addActivation(*layer1->getOutput(0), nvinfer1::ActivationType::kSIGMOID); // 注意更严谨的写法是*(layer1->getOutput(0)) 即对getOutput返回的指针进行解引用
    
    // 将我们需要的prob标记为输出
    network->markOutput(*prob->getOutput(0));

    printf("Workspace Size = 256 MB\n");
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 28);


    // ----------------------------- 3. 生成engine模型文件 -----------------------------
    //TensorRT 7.1.0版本已弃用buildCudaEngine方法，统一使用buildEngineWithConfig方法
    // nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    // if(engine == nullptr){
    //     printf("Build engine failed.\n");
    //     return -1;
    // }
// 注意：buildEngineWithConfig 在 10.x 也被弃用，建议改用 buildSerializedNetwork
    auto model_data = builder->buildSerializedNetwork(*network, *config);
    if (model_data == nullptr) {
        printf("Build serialized network failed.\n");
        return -1;
    }
    // ----------------------------- 4. 序列化模型文件并存储 -----------------------------
    // 将模型序列化，并储存为文件
    FILE* f = fopen("engine.trtmodel", "wb");
    if (f) {
        fwrite(model_data->data(), 1, model_data->size(), f);
        fclose(f);
    }
    else{printf("Error: Could not open file for writing!\n");}


// 【关键修改 4】：资源释放逻辑
    // 10.x 彻底删除了 .destroy() 方法，直接使用 delete 或智能指针
    delete model_data;
    delete network;
    delete config;
    delete builder;
    printf("Done.\n");
    return 0;
}