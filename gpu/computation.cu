//
// Created by david on 29/10/24.
//

#include "computation.cuh"

__global__ void getColorSimilarityMeasures(Pixel* bgData, Pixel* imgData, Pixel* similarityMeasuresData, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int i = y * width + x;

        float red_img = imgData[i][0];
        float green_img = imgData[i][1];
        float blue_img = imgData[i][2];

        float red_bg = bgData[i][0];
        float green_bg = bgData[i][1];
        float blue_bg = bgData[i][2];

        similarityMeasuresData[i][0] = fminf(red_img, red_bg) / fmaxf(red_img, red_bg);
        similarityMeasuresData[i][1] = fminf(green_img, green_bg) / fmaxf(green_img, green_bg);
        similarityMeasuresData[i][2] = fminf(blue_img, blue_bg) / fmaxf(blue_img, blue_bg);
    }
}

__device__ float getGrayscale(const Pixel& pixel)
{
    return 0.299f * pixel[0] + 0.587f * pixel[1] + 0.114f * pixel[2];
}

__global__ void calculateBitVector(const Pixel* imageData, uint8_t* bitVectorData, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int i = y * width + x;
        
        uint8_t texFeaturesVec = 0;

        float grayScale = getGrayscale(imageData[i]);

        auto isBorder = [&](int dx, int dy) {
            if (x + dx < 0 || x + dx >= width || y + dy < 0 || y + dy >= height)
                return 255.f;
            else
                return getGrayscale(imageData[(x + dx) * width + (y + dy)]);
        };

        texFeaturesVec = (texFeaturesVec << 1) + (isBorder(-1, -1) < grayScale);
        texFeaturesVec = (texFeaturesVec << 1) + (isBorder(0, -1) < grayScale);
        texFeaturesVec = (texFeaturesVec << 1) + (isBorder(1, -1) < grayScale);
        texFeaturesVec = (texFeaturesVec << 1) + (isBorder(1, 0) < grayScale);
        texFeaturesVec = (texFeaturesVec << 1) + (isBorder(1, 1) < grayScale);
        texFeaturesVec = (texFeaturesVec << 1) + (isBorder(0, 1) < grayScale);
        texFeaturesVec = (texFeaturesVec << 1) + (isBorder(-1, 1) < grayScale);
        texFeaturesVec = (texFeaturesVec << 1) + (isBorder(-1, 0) < grayScale);

        bitVectorData[i] = texFeaturesVec;
    }
}

__global__ void calculateTextureComponents(uint8_t* image, uint8_t* background, float* result, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int i = y * width + x;
        uint8_t similarity = ~(image[i] ^ background[i]);
        result[i] = __popc(similarity) / 8.0f;
    }
}

template <typename T>
__device__ void swap(T& a, T& b)
{
    T tmp = a;
    a = b;
    b = tmp;
}

__global__ void calculateChoquetIntegral(const Pixel* colorComponents, const float* textureComponents, float* result, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int i = y * width + x;

        float indicators[3] = {colorComponents[i][0], colorComponents[i][1], textureComponents[i]};

        if (indicators[0] > indicators[1])
            swap(indicators[0], indicators[1]);
        if (indicators[0] > indicators[2])
            swap(indicators[0], indicators[2]);
        if (indicators[1] > indicators[2])
            swap(indicators[1], indicators[2]);

        result[i] = indicators[0] * 0.1f + indicators[1] * 0.3f + indicators[2] * 0.6f;
    }
}

__global__ void calculateMask(const float* choquetIntegral, bool* result, int width, int height, float threshold)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int i = y * width + x;
        result[i] = choquetIntegral[i] > threshold ? false : true;
    }
}