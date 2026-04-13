#include <vector>   // 解决 vector 报错
#include <string>   // 解决 string 报错
#include <fstream>  // 解决 ifstream 报错
#include <math.h>   // 解决 exp 报错
// tensorRT include
#include <NvInfer.h>
#include <NvInferRuntime.h>

// cuda include
#include <cuda_runtime.h>

// system include
#include <stdio.h>

#include <iostream> 
#include <fstream> // 后面要用到ios这个库
#include <vector>
using namespace std;
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

bool build_model(){
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
        conv(3x3, pad=1)  input = 1, output = 1, bias = True     w=[[1.0, 2.0, 0.5], [0.1, 0.2, 0.5], [0.2, 0.2, 0.1]], b=0.0
          |
        relu
          |
        prob
    */

    // ----------------------------- 2. 输入，模型结构和输出的基本信息 -----------------------------
    const int num_input = 1;   // in_channel
    const int num_output = 1;  // out_channel
    float layer1_weight_values[] = {
        1.0, 2.0, 3.1, 
        0.1, 0.1, 0.1, 
        0.2, 0.2, 0.2
    };
    float layer1_bias_values[]   = {0.0};

    //输入指定数据的名称、数据类型和完整维度，将输入层添加到网络
    // 输入层：N, C, H, W
    // 如果要使用动态shape，必须让NetworkDefinition的维度定义为-1，in_channel是固定的
    auto input = network->addInput("image", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4(-1, num_input, -1, -1));
    nvinfer1::Weights layer1_weight = make_weights(layer1_weight_values, 9);
    nvinfer1::Weights layer1_bias   = make_weights(layer1_bias_values, 1);
    //添加全连接层
    auto layer1 = network->addConvolutionNd(*input, num_output, nvinfer1::Dims2(3, 3), layer1_weight, layer1_bias);
    layer1->setPaddingNd(nvinfer1::DimsHW(1, 1));
     //添加激活层 
    auto prob_layer = network->addActivation(*layer1->getOutput(0), nvinfer1::ActivationType::kRELU);
    // 【关键修改点】：给输出张量起名字
    // prob_layer->getOutput(0) 获取的是一个 ITensor* 指针
    auto output_tensor = prob_layer->getOutput(0);
    output_tensor->setName("prob"); // 这里的名字必须和推理端的 setOutputTensorAddress 一致
    // 将我们需要的prob标记为输出
    network->markOutput(*output_tensor);

    printf("Workspace Size = 256 MB\n");
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 28);

    // --------------------------------- 2.1 关于profile ----------------------------------
        // 10.x 强制要求动态输入必须配置profile，静态输入可以不配置profile但建议也配置以免误操作
        // profile的作用是指定动态输入的合法范围，超出这个范围的输入将无法推理
        // profile的数量没有限制，如果模型有多个输入，则必须多个profile
    int maxBatchSize = 10;
    // 如果模型有多个输入，则必须多个profile
    auto profile = builder->createOptimizationProfile();

    // 配置最小允许1 x 1 x 3 x 3
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, num_input, 3, 3));
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, num_input, 3, 3));

    // 配置最大允许10 x 1 x 5 x 5
    // if networkDims.d[i] != -1, then minDims.d[i] == optDims.d[i] == maxDims.d[i] == networkDims.d[i]
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(maxBatchSize, num_input, 5, 5));
    config->addOptimizationProfile(profile);

    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    if(engine == nullptr){
        printf("Build engine failed.\n");
        return false;
    }


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
        return false;
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
    delete engine;
    delete network;
    delete config;
    delete builder;
    printf("Done.\n");
    return true;
}


std::vector<unsigned char> load_file(const std::string& file){
    std::ifstream in(file, ios::in | ios::binary);
    if (!in.is_open())
        return {};

    in.seekg(0, ios::end);
    size_t length = in.tellg();

    std::vector<uint8_t> data;
    if (length > 0){
        in.seekg(0, ios::beg);
        data.resize(length);

        in.read((char*)&data[0], length);
    }
    in.close();
    return data;
}

void inference(){

    // ------------------------------ 1. 准备模型并加载   ----------------------------
    TRTLogger logger;
    auto engine_data = load_file("engine.trtmodel");
    // 执行推理前，需要创建一个推理的runtime接口实例。与builer一样，runtime需要logger：
    nvinfer1::IRuntime* runtime   = nvinfer1::createInferRuntime(logger);
    // 将模型从读取到engine_data中，则可以对其进行反序列化以获得engine
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
    if(engine == nullptr){
        printf("Deserialize cuda engine failed.\n");
        delete runtime;
        return;
    }

    nvinfer1::IExecutionContext* execution_context = engine->createExecutionContext();
    cudaStream_t stream = nullptr;
    // 创建CUDA流，以确定这个batch的推理是独立的
    cudaStreamCreate(&stream);

    /*
        Network definition:

        image
          |
        conv(3x3, pad=1)  input = 1, output = 1, bias = True     w=[[1.0, 2.0, 0.5], [0.1, 0.2, 0.5], [0.2, 0.2, 0.1]], b=0.0
          |
        relu
          |
        prob
    */

    // ------------------------------ 2. 准备好要推理的数据并搬运到GPU   ----------------------------
    float input_data_host[] = {
        // batch 0
        1,   1,   1,
        1,   1,   1,
        1,   1,   1,

        // batch 1
        -1,   1,   1,
        1,   0,   1,
        1,   1,   -1
    };
    float* input_data_device = nullptr;

    // 3x3输入，对应3x3输出
    int ib = 2;
    int iw = 3;
    int ih = 3;

    float output_data_host[ib * iw * ih];
    float* output_data_device = nullptr;
    cudaMalloc(&input_data_device, sizeof(input_data_host));
    cudaMalloc(&output_data_device, sizeof(output_data_host));
    cudaMemcpyAsync(input_data_device, input_data_host, sizeof(input_data_host), cudaMemcpyHostToDevice, stream);


    // ------------------------------ 3. 推理并将结果搬运回CPU   ----------------------------
    execution_context->setInputShape("image", nvinfer1::Dims4(ib, 1, ih, iw));
    execution_context->setInputTensorAddress("image", input_data_device);
    execution_context->setOutputTensorAddress("prob", output_data_device);
    bool success      = execution_context->enqueueV3(stream);
    cudaMemcpyAsync(output_data_host, output_data_device, sizeof(output_data_host), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    printf("output_data_host = %f, %f\n", output_data_host[0], output_data_host[1]);
    // ------------------------------ 4. 手动推理进行验证 ----------------------------

    printf("手动验证计算结果：\n");
    for(int b = 0; b < ib; ++b){
        printf("batch %d. output_data_host = \n", b);
        for(int i = 0; i < iw * ih; ++i){
            printf("%f, ", output_data_host[b * iw * ih + i]);
            if((i + 1) % iw == 0)
                printf("\n");
        }
    }

    // ------------------------------ 5. 释放内存 ----------------------------
    printf("Clean memory\n");
    cudaStreamDestroy(stream);
    cudaFree(input_data_device);
    cudaFree(output_data_device);
    delete execution_context;
    delete engine;
    delete runtime;

}

int main(){

    if(!build_model()){
        return -1;
    }
    inference();
    return 0;
}

