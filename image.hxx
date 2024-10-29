//
// Created by david on 27/10/24.
//

#ifndef IMAGE_HXX
#define IMAGE_HXX

template<typename T>
Image<T>::Image(int width, int height) {
    width_ = width;
    height_ = height;
    data_.resize(width * height);
}

template<typename T>
Image<T>::~Image() {
    data_.clear();
}

template<typename T>
const T& Image<T>::at(int x, int y) const {
    return data_[y * width_ + x];
}

template <typename T>
void Image<T>::set(int x, int y, const T& value) {
    data_[y * width_ + x] = value;
}

template<typename T>
void Image<T>::save(const std::string& filename) const
{
    std::ofstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    file << "P5\n" << width_ << " " << height_ << "\n255\n";
    for (const T& pixel : data_)
    {
        unsigned char value = pixel ? 255 : 0;
        file.write(reinterpret_cast<const char*>(&value), 1);
    }

    file.close();
    if (!file)
        std::cerr << "Failed to write to file: " << filename << std::endl;
    else
        std::cout << "Image saved successfully: " << filename << std::endl;
}

#endif //IMAGE_HXX
