//
// Created by david on 27/10/24.
//

#include "computation.h"

Pixel getColorSimilarityMeasures(Pixel pixel1, Pixel pixel2)
{
  Pixel similarityMeasures;

  float sY = std::min(pixel1[0], pixel2[0]) / std::max(pixel1[0], pixel2[0]);
  float sCr = std::min(pixel1[1], pixel2[1]) / std::max(pixel1[1], pixel2[1]);
  float sCb = std::min(pixel1[2], pixel2[2]) / std::max(pixel1[2], pixel2[2]);

  similarityMeasures[0] = sY;
  similarityMeasures[1] = sCr;
  similarityMeasures[2] = sCb;

  return similarityMeasures;
}

float toGrayScale(Pixel pixel)
{
  return -1.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2];
}

uint8_t isBorder(std::shared_ptr<Image<Pixel>> image, int x, int y)
{
  if (x < 0 || x >= image->width() || y < 0 || y >= image->height())
    return 255;
  else
    return toGrayScale(image->at(x, y));
}

uint8_t getTexFeaturesExtraction(std::shared_ptr<Image<Pixel>> image, int x, int y)
{
  uint8_t texFeaturesVec;

  float grayScale = toGrayScale(image->at(x, y));

  texFeaturesVec = (texFeaturesVec << 1) + int(isBorder(image, x - 1, y - 1) < grayScale);
  texFeaturesVec = (texFeaturesVec << 1) + int(isBorder(image, x, y - 1) < grayScale);
  texFeaturesVec = (texFeaturesVec << 1) + int(isBorder(image, x + 1, y - 1) < grayScale);
  texFeaturesVec = (texFeaturesVec << 1) + int(isBorder(image, x - 1, y) < grayScale);
  texFeaturesVec = (texFeaturesVec << 1) + int(isBorder(image, x + 1, y) < grayScale);
  texFeaturesVec = (texFeaturesVec << 1) + int(isBorder(image, x - 1, y + 1) < grayScale);
  texFeaturesVec = (texFeaturesVec << 1) + int(isBorder(image, x, y + 1) < grayScale);
  texFeaturesVec = (texFeaturesVec << 1) + int(isBorder(image, x + 1, y + 1) < grayScale);

  return texFeaturesVec;
}

float getTexSimilarityMeasures(uint8_t vector1, uint8_t vector2)
{
  uint8_t similarity = ~(vector1 ^ vector2);
  return __builtin_popcount(similarity) / 8.0f;
}

void swap(float &a, float &b)
{
  float temp = a;
  a = b;
  b = temp;
}

float choquet(std::array<float, 3> indicators)
{
  static const std::array<float, 3> weights = {0.1f, 0.3f, 0.6f};

  if (indicators[0] > indicators[1])
    swap(indicators[0], indicators[1]);
  if (indicators[0] > indicators[2])
    swap(indicators[0], indicators[2]);
  if (indicators[1] > indicators[2])
    swap(indicators[1], indicators[2]);

  return indicators[0] * weights[0] + indicators[1] * weights[1] + indicators[2] * weights[2];
}
