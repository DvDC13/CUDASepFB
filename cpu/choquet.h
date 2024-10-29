//
// Created by david on 28/10/24.
//

#ifndef CHOQUET_H
#define CHOQUET_H

#include "computation.h"

std::shared_ptr<Image<bool>> computeChoquet(std::shared_ptr<Image<Pixel>> background, std::shared_ptr<Image<Pixel>> image);

#endif //CHOQUET_H
