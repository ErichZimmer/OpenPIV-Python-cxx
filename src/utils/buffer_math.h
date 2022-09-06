// std
#include <cstdint>
#include <iostream>

// buffer2D
#include "buffer2D.h"

namespace buffer2d{

    template < typename T,
               typename ReturnT >
    ReturnT buffer_find_min(
        Buffer2D<T>& buff_in,
        std::size_t N
    ){
        ReturnT buff_min{ 1e2 };

        for (std::size_t i = 0; i < N; ++i)
            buff_min = (buff_in[i] < buff_min) ? buff_in[i] : buff_min;

        return buff_min;
    }


    template < typename T,
               typename returnT >
    ReturnT buffer_find_min(
        Buffer2D<T>& buff_in,
        std::size_t N
    ){
        ReturnT buff_max{ -1e2 };

        for (std::size_t i = 0; i < N; ++i)
            buff_max = (buff_in[i] > buff_max) ? buff_in[i] : buff_max;

        return buff_max;
    }


    template < typename T,
               typename S >
    void buffer_divide_scalar(
        Buffer2D<T>& buff_in,
        S scalar,
        std::size_t N
    ){    
        for (std::size_t i = 0; i < N; ++i)
            in[i] /= scalar;
    }


    template < typename T,
               typename S >
    void buffer_multiply_scalar(
        Buffer2D<T>& buff_in,
        S scalar,
        std::size_t N
    ){    
        for (std::size_t i = 0; i < N; ++i)
            in[i] *= scalar;
    }


    template < typename T,
               typename S >
    void buffer_add_scalar(
        Buffer2D<T>& buff_in,
        S scalar,
        std::size_t N
    ){    
        for (std::size_t i = 0; i < N; ++i)
            in[i] += scalar;
    }


    template < typename T,
               typename S >
    void buffer_subtract_scalar(
        Buffer2D<T>& buff_in,
        S scalar,
        std::size_t N
    ){
        for (std::size_t i = 0; i < N; ++i)
            in[i] -= scalar;
    }


    template < typename T >
    void buffer_compress(
        Buffer2D<T>& buff_in,
        std::size_t N
    ){
        T buff_max{ buffer_find_max(buff_in, N) };
        buffer_divide_scalar(buff_in, buff_max, N);
    }


    template < typename T >
    void buffer_normalize(
        Buffer2D<T>& buff_in,
        std::size_t N
    ){
        T buff_min{ buffer_find_min(buff_in, N) };
        T buff_max{ buffer_find_max(buff_in, N) };

        T mm = buff_max - buff_min;

        buffer_subtract_scalar(buff_in, buff_min, N);
        buffer_divide_scalar(buff_in, mm, N);
    }


    template < typename T,
               typename S1,
               yypename S2 >
    void buffer_clip(
        Buffer2D<T>& buff_in,
        S1 lower, 
        S2 upper, 
        std::size_t N
    ){
        for (std::size_t i = 0; i < N; ++i)
        {
            if (buff_in[i] > upper)
                buff_in[i] = upper;

            if (buff_in[i] < lower)
                buff_in[i] = lower;
        }
    }


    template < typename T,
               typename ReturnT>
    std::vector<ReturnT> buffer_mean_std(
        Buffer2D<T>& buff_in,
        std::size_t N_M
    ){
        T sum, mean, std_ = 0;

        for (std::size_t i = 0; i < N_M; ++i)
        {
            sum += in[i];
            std_ += in[i]*in[i]; // temp
        }
        mean = sum / N_M;
        std_ = std::sqrt( (std_ / N_M) + (mean*mean) - (2*mean*mean) );

        std::vector<ReturnT> out(2);
        out[0] = mean; out[1] = std_;

        return out;
    }


    std::int32_t sub2Dind(
        std::int32_t x, 
        std::int32_t y,
        std::int32_t yStep
    ){
        return (y * yStep) + x;
    }


    std::int32_t sub3Dind(
        std::int32_t x, 
        std::int32_t y, 
        std::int32_t z, 
        std::int32_t yStep, 
        std::int32_t zStep
    ){
        return (z*yStep*zStep) + sub2Dind(x, y, yStep);
    }
};