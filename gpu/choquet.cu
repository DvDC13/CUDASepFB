//
// Created by david on 29/10/24.
//

#include "choquet.cuh"

std::shared_ptr<Image<uint8_t>> getBitVector(std::shared_ptr<Image<Pixel>> image)
{
    int width = image->width();
    int height = image->height();

    Pixel* d_ImageData;
    gpuErrorCheck(cudaMalloc(&d_ImageData, width * height * sizeof(Pixel)));
    gpuErrorCheck(cudaMemcpy(d_ImageData, image->data().data(), width * height * sizeof(Pixel), cudaMemcpyHostToDevice));

    uint8_t* d_imageBitVector;
    gpuErrorCheck(cudaMalloc(&d_imageBitVector, width * height * sizeof(uint8_t)));

    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    calculateBitVector<<<grid, block>>>(d_ImageData, d_imageBitVector, width, height);
    cudaDeviceSynchronize();

    std::vector<uint8_t> h_imageBitVector(width * height);
    gpuErrorCheck(cudaMemcpy(h_imageBitVector.data(), d_imageBitVector, width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost));

    gpuErrorCheck(cudaFree(d_ImageData));
    gpuErrorCheck(cudaFree(d_imageBitVector));

    std::shared_ptr<Image<uint8_t>> result = std::make_shared<Image<uint8_t>>(width, height);
    result->set_data(h_imageBitVector);

    return result;
}

std::shared_ptr<Image<Pixel>> getColorImage(std::shared_ptr<Image<Pixel>> image, std::shared_ptr<Image<Pixel>> background)
{
    int width = image->width();
    int height = image->height();

    Pixel* d_ImageData;
    Pixel* d_BackgroundData;
    Pixel* d_colorComponents;

    gpuErrorCheck(cudaMalloc(&d_ImageData, width * height * sizeof(Pixel)));
    gpuErrorCheck(cudaMalloc(&d_BackgroundData, width * height * sizeof(Pixel)));
    gpuErrorCheck(cudaMalloc(&d_colorComponents, width * height * sizeof(Pixel)));

    gpuErrorCheck(cudaMemcpy(d_ImageData, image->data().data(), width * height * sizeof(Pixel), cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(d_BackgroundData, background->data().data(), width * height * sizeof(Pixel), cudaMemcpyHostToDevice));

    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    getColorSimilarityMeasures<<<grid, block>>>(d_BackgroundData, d_ImageData, d_colorComponents, width, height);
    cudaDeviceSynchronize();

    std::vector<Pixel> h_colorComponents(width * height);
    gpuErrorCheck(cudaMemcpy(h_colorComponents.data(), d_colorComponents, width * height * sizeof(Pixel), cudaMemcpyDeviceToHost));

    gpuErrorCheck(cudaFree(d_ImageData));
    gpuErrorCheck(cudaFree(d_BackgroundData));
    gpuErrorCheck(cudaFree(d_colorComponents));

    std::shared_ptr<Image<Pixel>> result = std::make_shared<Image<Pixel>>(width, height);
    result->set_data(h_colorComponents);

    return result;
}

std::shared_ptr<Image<float>> computeTextureComponents(std::shared_ptr<Image<uint8_t>> imageBitVector, std::shared_ptr<Image<uint8_t>> backgroundBitVector)
{
    int width = imageBitVector->width();
    int height = imageBitVector->height();

    uint8_t* d_ImageData;
    uint8_t* d_BackgroundData;
    float* d_textureComponents;

    gpuErrorCheck(cudaMalloc(&d_ImageData, width * height * sizeof(uint8_t)));
    gpuErrorCheck(cudaMalloc(&d_BackgroundData, width * height * sizeof(uint8_t)));
    gpuErrorCheck(cudaMalloc(&d_textureComponents, width * height * sizeof(float)));

    gpuErrorCheck(cudaMemcpy(d_ImageData, imageBitVector->data().data(), width * height * sizeof(uint8_t), cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(d_BackgroundData, backgroundBitVector->data().data(), width * height * sizeof(uint8_t), cudaMemcpyHostToDevice));

    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    calculateTextureComponents<<<grid, block>>>(d_ImageData, d_BackgroundData, d_textureComponents, width, height);
    cudaDeviceSynchronize();

    std::vector<float> h_textureComponents(width * height);
    gpuErrorCheck(cudaMemcpy(h_textureComponents.data(), d_textureComponents, width * height * sizeof(float), cudaMemcpyDeviceToHost));

    gpuErrorCheck(cudaFree(d_ImageData));
    gpuErrorCheck(cudaFree(d_BackgroundData));
    gpuErrorCheck(cudaFree(d_textureComponents));

    std::shared_ptr<Image<float>> result = std::make_shared<Image<float>>(width, height);
    result->set_data(h_textureComponents);

    return result;
}

std::shared_ptr<Image<float>> computeChoquetIntegral(std::shared_ptr<Image<Pixel>> colorComponents, std::shared_ptr<Image<float>> textureComponents)
{
    int width = colorComponents->width();
    int height = colorComponents->height();

    Pixel* d_colorComponents;
    float* d_textureComponents;
    float* d_choquetIntegral;

    gpuErrorCheck(cudaMalloc(&d_colorComponents, width * height * sizeof(Pixel)));
    gpuErrorCheck(cudaMalloc(&d_textureComponents, width * height * sizeof(float)));
    gpuErrorCheck(cudaMalloc(&d_choquetIntegral, width * height * sizeof(float)));

    gpuErrorCheck(cudaMemcpy(d_colorComponents, colorComponents->data().data(), width * height * sizeof(Pixel), cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(d_textureComponents, textureComponents->data().data(), width * height * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    calculateChoquetIntegral<<<grid, block>>>(d_colorComponents, d_textureComponents, d_choquetIntegral, width, height);
    cudaDeviceSynchronize();

    std::vector<float> h_choquetIntegral(width * height);
    gpuErrorCheck(cudaMemcpy(h_choquetIntegral.data(), d_choquetIntegral, width * height * sizeof(float), cudaMemcpyDeviceToHost));

    gpuErrorCheck(cudaFree(d_colorComponents));
    gpuErrorCheck(cudaFree(d_textureComponents));
    gpuErrorCheck(cudaFree(d_choquetIntegral));

    std::shared_ptr<Image<float>> result = std::make_shared<Image<float>>(width, height);
    result->set_data(h_choquetIntegral);

    return result;
}

std::shared_ptr<Image<bool>> computeMask(std::shared_ptr<Image<float>> choquetIntegral, float threshold)
{
    int width = choquetIntegral->width();
    int height = choquetIntegral->height();

    float* d_choquetIntegral;
    bool* d_result;

    gpuErrorCheck(cudaMalloc(&d_choquetIntegral, width * height * sizeof(float)));
    gpuErrorCheck(cudaMalloc(&d_result, width * height * sizeof(bool)));

    gpuErrorCheck(cudaMemcpy(d_choquetIntegral, choquetIntegral->data().data(), width * height * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    calculateMask<<<grid, block>>>(d_choquetIntegral, d_result, width, height, threshold);
    cudaDeviceSynchronize();

    bool* h_result = new bool[width * height];
    gpuErrorCheck(cudaMemcpy(h_result, d_result, width * height * sizeof(bool), cudaMemcpyDeviceToHost));

    gpuErrorCheck(cudaFree(d_choquetIntegral));
    gpuErrorCheck(cudaFree(d_result));

    std::shared_ptr<Image<bool>> result = std::make_shared<Image<bool>>(width, height);
    result->set_data(h_result);

    return result;
}

std::shared_ptr<Image<bool>> computeChoquet_gpu(std::shared_ptr<Image<Pixel>> background, std::shared_ptr<Image<Pixel>> image)
{
    static std::shared_ptr<Image<uint8_t>> bgBitVector = getBitVector(background);
    std::shared_ptr<Image<uint8_t>> imageBitVector = getBitVector(image);

    std::shared_ptr<Image<Pixel>> colorComponents = getColorImage(image, background);

    std::shared_ptr<Image<float>> textureComponents = computeTextureComponents(imageBitVector, bgBitVector);

    std::shared_ptr<Image<float>> choquetIntegral = computeChoquetIntegral(colorComponents, textureComponents);

    std::shared_ptr<Image<bool>> result = computeMask(choquetIntegral, 0.67f);

    return result;
}
