//
// Created by david on 29/10/24.
//

#ifndef CHOQUET_CUH
#define CHOQUET_CUH

#include "computation.cuh"

std::shared_ptr<Image<bool>> computeChoquet_gpu(std::shared_ptr<Image<Pixel>> background, std::shared_ptr<Image<Pixel>> image);

#endif //CHOQUET_CUH