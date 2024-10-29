//
// Created by david on 27/10/24.
//

#ifndef COMPUTATION_H
#define COMPUTATION_H

#include "constants.h"
#include "image.h"

Pixel getColorSimilarityMeasures(Pixel pixel1, Pixel pixel2);

uint8_t getTexFeaturesExtraction(std::shared_ptr<Image<Pixel>> image, int x, int y);

float getTexSimilarityMeasures(uint8_t vector1, uint8_t vector2);

float choquet(std::array<float, 3> indicators);

#endif //COMPUTATION_H