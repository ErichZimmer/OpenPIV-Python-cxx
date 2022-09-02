c++ implementation of Taylor Expansion via finite differences based on fast_interp by David Stein. This implementations difers from the original as it is written to perform image mapping for window deformations, hence the need to rewrite the library. 

The original source code can be found at:
- https://github.com/dbstein/fast_interp

This library has been rewritten from the Numba accelerated Python version into c++ and provides the following modifications:
- map coordinates by interpolation following the format f(Z, Xq, Yq)

Contents:
- include/taylor_expansion.h : Header
- src/taylor_expansion.cpp : Source
- wrapper.cpp : wrapper interface to module
- CMakeLists.txt : Used to compile module
- README : This file

Copyright Statement:
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.


