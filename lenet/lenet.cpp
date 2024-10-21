#include <chrono>
#include <filesystem>
#include <map>
#include <numeric>
#include <vector>
#include "logging.h"
#include "utils.h"

// parameters we know about the lenet-5
#define INPUT_H 32
#define INPUT_W 32
#define INPUT_SIZE (INPUT_H * INPUT_W)
#define OUTPUT_SIZE 10
#define INPUT_NAME "data"
#define OUTPUT_NAME "prob"

using namespace nvinfer1;

static Logger gLogger;

/**
 * @brief Creat the engine using only the API and not any parser.
 *
 * @param N max batch size
 * @param runtime runtime
 * @param builder builder
 * @param config config
 * @param dt data type
 * @return ICudaEngine*
 */
ICudaEngine* createLenetEngine(int32_t N, IRuntime* runtime, IBuilder* builder, IBuilderConfig* config, DataType dt) {
    INetworkDefinition* network = builder->createNetworkV2(1u);

    // Create input tensor of shape { 1, 1, 32, 32 } with name INPUT_NAME
    ITensor* data = network->addInput(INPUT_NAME, dt, Dims4{N, 1, INPUT_H, INPUT_W});
    assert(data);

    // clang-format off
    // Add convolution layer with 6 outputs and a 5x5 filter.
    std::map<std::string, Weights> weightMap = loadWeights("lenet5.wts");
    IConvolutionLayer* conv1 = network->addConvolutionNd(*data, 6, DimsHW{5, 5}, weightMap["conv1.weight"], weightMap["conv1.bias"]);
    assert(conv1);
    conv1->setStrideNd(DimsHW{1, 1});
    conv1->setName("conv1");

    // Add activation layer using the ReLU algorithm.
    IActivationLayer* relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
    assert(relu1);
    relu1->setName("relu1");

    // Add max pooling layer with stride of 2x2 and kernel size of 2x2.
    IPoolingLayer* pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kAVERAGE, DimsHW{2, 2});
    assert(pool1);
    pool1->setStrideNd(DimsHW{2, 2});
    pool1->setName("pool1");

    // Add second convolution layer with 16 outputs and a 5x5 filter.
    IConvolutionLayer* conv2 = network->addConvolutionNd(*pool1->getOutput(0), 16, DimsHW{5, 5}, weightMap["conv2.weight"], weightMap["conv2.bias"]);
    assert(conv2);
    conv2->setStrideNd(DimsHW{1, 1});
    conv2->setName("conv2");

    // Add activation layer using the ReLU algorithm.
    IActivationLayer* relu2 = network->addActivation(*conv2->getOutput(0), ActivationType::kRELU);
    assert(relu2);

    // Add second max pooling layer with stride of 2x2 and kernel size of 2x2>
    IPoolingLayer* pool2 = network->addPoolingNd(*relu2->getOutput(0), PoolingType::kAVERAGE, DimsHW{2, 2});
    assert(pool2);
    pool2->setStrideNd(DimsHW{2, 2});
    pool2->setName("pool2");

    // Add fully connected layer
    auto* flatten = network->addShuffle(*pool2->getOutput(0));
    flatten->setReshapeDimensions(Dims2{-1, 400});
    auto* tensor_fc1w = network->addConstant(Dims2{120, 400}, weightMap["fc1.weight"])->getOutput(0);
    auto* fc1w = network->addMatrixMultiply(*tensor_fc1w, MatrixOperation::kNONE, *flatten->getOutput(0), MatrixOperation::kTRANSPOSE);
    assert(tensor_fc1w && fc1w);
    auto tensor_fc1b = network->addConstant(Dims2{120, 1}, weightMap["fc1.bias"])->getOutput(0);
    auto* fc1b = network->addElementWise(*fc1w->getOutput(0), *tensor_fc1b, ElementWiseOperation::kSUM);
    fc1b->setName("fc1b");
    assert(tensor_fc1b && fc1b);

    // Add activation layer using the ReLU algorithm.
    IActivationLayer* relu3 = network->addActivation(*fc1b->getOutput(0), ActivationType::kRELU);
    assert(relu3);
    auto* flatten_relu3 = network->addShuffle(*relu3->getOutput(0));
    flatten_relu3->setReshapeDimensions(Dims2{-1, 120});

    // Add second fully connected layer
    auto* tensor_fc2w = network->addConstant(Dims2{84, 120}, weightMap["fc2.weight"])->getOutput(0);
    auto* fc2w = network->addMatrixMultiply(*tensor_fc2w, MatrixOperation::kNONE, *flatten_relu3->getOutput(0), MatrixOperation::kTRANSPOSE);
    assert(tensor_fc2w && fc2w);
    fc2w->setName("fc2w");
    auto* tensor_fc2b = network->addConstant(Dims2{84, 1}, weightMap["fc2.bias"])->getOutput(0);
    auto* fc2b = network->addElementWise(*fc2w->getOutput(0), *tensor_fc2b, ElementWiseOperation::kSUM);
    assert(tensor_fc2b && fc2b);
    fc2b->setName("fc2b");

    // Add activation layer using the ReLU algorithm.
    IActivationLayer* relu4 = network->addActivation(*fc2b->getOutput(0), ActivationType::kRELU);
    assert(relu4);
    auto* flatten_relu4 = network->addShuffle(*relu4->getOutput(0));
    flatten_relu4->setReshapeDimensions(Dims2{-1, 84});

    // Add third fully connected layer
    auto* tensor_fc3w = network->addConstant(Dims2{10, 84}, weightMap["fc3.weight"])->getOutput(0);
    auto* fc3w = network->addMatrixMultiply(*tensor_fc3w, MatrixOperation::kNONE, *flatten_relu4->getOutput(0), MatrixOperation::kTRANSPOSE);
    assert(tensor_fc3w && fc3w);
    fc3w->setName("fc3w");
    auto* tensor_fc3b = network->addConstant(Dims2{10, 1}, weightMap["fc3.bias"])->getOutput(0);
    auto* fc3b = network->addElementWise(*fc3w->getOutput(0), *tensor_fc3b, ElementWiseOperation::kSUM);
    assert(tensor_fc3b && fc3b);
    fc3b->setName("fc3b");
    // clang-format on

    // Add softmax layer to determine the probability.
    ISoftMaxLayer* prob = network->addSoftMax(*fc3b->getOutput(0));
    assert(prob);
    prob->getOutput(0)->setName(OUTPUT_NAME);
    network->markOutput(*prob->getOutput(0));

#if TRT_VERSION >= 8400
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, WORKSPACE_SIZE);
#else
    config->setMaxWorkspaceSize(WORKSPACE_SIZE);
    builder->setMaxBatchSize(maxBatchSize);
#endif

    // Build engine
#if TRT_VERSION >= 8000
    IHostMemory* serialized_mem = builder->buildSerializedNetwork(*network, *config);
    ICudaEngine* engine = runtime->deserializeCudaEngine(serialized_mem->data(), serialized_mem->size());
#else
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
#endif

#if TRT_VERSION >= 8000
    delete network;
#else
    network->destroy();
#endif

    // Release host memory
    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }

    return engine;
}

/**
 * @brief create a model using the API directly and serialize it to a stream
 *
 * @param N max batch size
 * @param runtime runtime
 * @param modelStream
 */
void APIToModel(int32_t N, IRuntime* runtime, IHostMemory** modelStream) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createLenetEngine(N, runtime, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

#if TRT_VERSION >= 8000
    delete engine;
    delete config;
    delete builder;
#else
    engine->destroy();
    config->destroy();
    builder->destroy();
#endif
}

void doInference(IExecutionContext& ctx, float* input, float* output, int batchSize) {
    const ICudaEngine& engine = ctx.getEngine();

    // Find input/output index so we can bind them to the buffers we provide later
#if TRT_VERSION >= 8000
    int32_t nIO = engine.getNbIOTensors();
    const int inputIndex = 0;
    const int outputIndex = engine.getNbIOTensors() - 1;
#else
    int32_t nIO = engine.getNbBindings();
    const int inputIndex = engine.getBindingIndex(INPUT_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_NAME);
#endif
    assert(nIO == 2);  // lenet-5 contains 1 input and 1 output

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // Pointers to input and output cuda buffers to pass to engine
    // Note that indices are guaranteed to be less than total I/O number
    std::vector<void*> buffers(nIO);
    CHECK(cudaMallocAsync(&buffers[inputIndex], batchSize * INPUT_SIZE * sizeof(float), stream));
    CHECK(cudaMallocAsync(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), stream));
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice,
                          stream));

    // Run inference
#if TRT_VERSION >= 8000
    for (int32_t i = 0; i < engine.getNbIOTensors(); i++) {
        auto const name = engine.getIOTensorName(i);
        auto dims = ctx.getTensorShape(name);
        auto total = std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int>());
        std::cout << name << " with total element size: " << total << std::endl;
        ctx.setTensorAddress(name, buffers[i]);
    }
    assert(ctx.enqueueV3(stream));
#else
    assert(ctx.enqueueV2(buffers.data(), stream, nullptr));
#endif
    // Use async API so that no synchronization is needed
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost,
                          stream));
    for (auto& buffer : buffers) {
        CHECK(cudaFreeAsync(buffer, stream));
    }
    CHECK(cudaStreamDestroy(stream));
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./lenet -s   // serialize model to plan file" << std::endl;
        std::cerr << "./lenet -d   // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);

    char* trtModelStream{nullptr};
    size_t size{0};

    if (std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(1, runtime, &modelStream);
        assert(modelStream != nullptr);

        std::ofstream p("lenet5.engine", std::ios::binary | std::ios::trunc);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());

#if TRT_VERSION >= 8000
        delete modelStream;
#else
        modelStream->destroy();
#endif
        std::cout << "serialized weights to lenet5.engine" << std::endl;
        return 0;
    } else if (std::string(argv[1]) == "-d") {
        std::ifstream file("lenet5.engine", std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    } else {
        return -1;
    }

    // Mock data. Align with python demo if you need to validate accuracy
    std::vector<float> data(INPUT_H * INPUT_W, 1.f);

#if TRT_VERSION >= 8000
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
#else
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
#endif
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    // Run inference
    float prob[OUTPUT_SIZE];
    for (int i = 0; i < 1000; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        doInference(*context, data.data(), prob, 1);
        auto end = std::chrono::high_resolution_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "execution time: " << dur << "us" << std::endl;
    }

#if TRT_VERSION >= 8000
    delete context;
    delete engine;
    delete runtime;
#else
    context->destroy();
    engine->destroy();
    runtime->destroy();
#endif

    // Print histogram of the output distribution
    std::cout << "\nOutput:\n\n";
    for (unsigned int i = 0; i < 10; i++) {
        std::cout << prob[i] << ", " << std::flush;
    }
    std::cout << std::endl;

    return 0;
}
