//
// Created by david on 29/10/24.
//

#ifndef COMPUTATION_CUH
#define COMPUTATION_CUH

#include "error.cuh"
#include "constants.h"
#include "image.h"

__global__ void getColorSimilarityMeasures(Pixel* bgData, Pixel* imgData, Pixel* similarityMeasuresData, int width, int height);

__device__ float getGrayscale(const Pixel& pixel);

__global__ void calculateBitVector(const Pixel* imageData, uint8_t* bitVectorData, int width, int height);

__global__ void calculateTextureComponents(uint8_t* image, uint8_t* background, float* result, int width, int height);

__global__ void calculateChoquetIntegral(const Pixel* colorComponents, const float* textureComponents, float* result, int width, int height);

__global__ void calculateMask(const float* choquetIntegral, bool* result, int width, int height, float threshold);

#endif //COMPUTATION_CUH