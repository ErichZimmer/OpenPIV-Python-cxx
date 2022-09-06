// std
#include <array>
#include <cstdint>
#include <exception>
#include <iostream>

namespace buffer2d{

    template < typename T >
    class Buffer2D
    {
        public:
          Buffer2D(size_t rows, size_t cols, T* data) : m_rows(rows), m_cols(cols), m_step(cols), m_data(data){};

            // private members
            T* data() { return &m_data; }
            size_t rows() { return m_rows; }
            size_t cols() { return m_cols; }
            size_t step() { return m_step; }
            size_t size() { return m_rows * m_cols; }

            // assignment
            #ifdef DEBUG 
                // linear index
                inline T& operator[](size_t i)
                {
                    if ( (i > m_cols * m_rows) || (i < 0 ) )
                       throw std::runtime_error("Index out of range");
                    return m_data[i]; 
                }

                // 2D index
                inline T& operator()(size_t i, size_t j)
                {
                    if ( (i > m_rows) || (j > m_cols ) ||  (i < 0) || (j < 0 ) )
                       throw std::runtime_error("Index out of range");
                    return m_data[i * m_step + j];
                }
            #else
                // linear index
                inline T& operator[](size_t i) { return m_data[i]; }

                // 2D index
                inline T& operator()(size_t i, size_t j) { return m_data[i * m_step + j]; }
            #endif

            // linear index
            inline const T& operator[](size_t i) const { return const_cast<Buffer2D*>(this)->operator[](i); }

            // 2D index
            inline const T& operator()(size_t i, size_t j) const
            {
                return const_cast<Buffer2D*>(this)->operator()(i, j);
            }

            // line index
            inline T* line(size_t i)
            {
                if ( (i > m_rows) || (i < 0) )
                    throw std::runtime_error("Line out of index");

                return &m_data[i*m_cols];
            }
            inline const T* line( size_t i ) const { return const_cast<Buffer2D*>(this)->line(i); }

        private:
            size_t m_rows, m_cols, m_step;
            T* m_data;
    };

};