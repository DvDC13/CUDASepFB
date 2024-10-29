//
// Created by david on 27/10/24.
//

#ifndef IMAGE_CUH
#define IMAGE_CUH

#include <iostream>
#include <vector>
#include <fstream>

template<typename T>
class Image
{
public: // public member
    Image(int width, int height);
    ~Image();

    inline int width() const { return width_; }
    inline int height() const { return height_; }
    inline std::vector<T> data() const { return data_; }

    const T& at(int x, int y) const;
    T& operator()(int x, int y) { return at(x, y); }
    const T& operator()(int x, int y) const { return at(x, y); }

    void set(int x, int y, const T& value);

    void save(const std::string& filename) const;

private: // private member
    int width_;
    int height_;
    std::vector<T> data_;
};

#include "image.hxx"


#endif //IMAGE_CUH
