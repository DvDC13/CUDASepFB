//
// Created by david on 28/10/24.
//

#include "choquet.h"

std::shared_ptr<Image<uint8_t>> getBitVector(std::shared_ptr<Image<Pixel>> image)
{
  std::shared_ptr<Image<uint8_t>> result = std::make_shared<Image<uint8_t>>(image->width(), image->height());

  for (int y = 0; y < image->height(); y++)
    for (int x = 0; x < image->width(); x++)
      result->set(x, y, getTexFeaturesExtraction(image, x, y));

  return result;
}

std::shared_ptr<Image<bool>> computeChoquet(std::shared_ptr<Image<Pixel>> background, std::shared_ptr<Image<Pixel>> image)
{
  int height = image->height();
  int width = image->width();

  static std::shared_ptr<Image<uint8_t>> bgBitVector = getBitVector(background);
  std::shared_ptr<Image<uint8_t>> imageBitVector = getBitVector(image);

  std::shared_ptr<Image<bool>> result = std::make_shared<Image<bool>>(width, height);

  for (int i = 0; i < height; i++)
  {
    for (int j = 0; j < width; j++)
    {
      // Color
      std::array<float, 3> colorRGB = getColorSimilarityMeasures(background->at(j, i), image->at(j, i));

      // Texture
      uint8_t vector1 = bgBitVector->at(j, i);
      uint8_t vector2 = imageBitVector->at(j, i);
      float texSimilarity = getTexSimilarityMeasures(vector1, vector2);

      // Choquet
      std::array<float, 3> choquetIndicators = {colorRGB[0], colorRGB[1], texSimilarity};
      float choquetSimilarity = choquet(choquetIndicators);

      const bool isForeground = (choquetSimilarity < 0.67f) ? true : false;
      result->set(j, i, isForeground);
    }
  }

  return result;
}