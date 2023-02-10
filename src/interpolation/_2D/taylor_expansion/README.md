c++ implementation of Taylor Expansion via finite differences based on fast_interp by David Stein. This implementations differs from the original as it is written to perform image mapping for window deformations, hence the need to rewrite the library. 

The original source code can be found at:
- https://github.com/dbstein/fast_interp

This library has been rewritten from the Numba accelerated Python version into c++ and provides the following modifications:
- map coordinates by interpolation following the format f(Z, Xq, Yq)

Contents:
- include/taylor_expansion.h : Header
- src/taylor_expansion.cpp : Source
- wrapper.cpp : wrapper interface to submodule
- CMakeLists.txt : Used to compile submodule
- LICENSE.txt : License of this submodule
- README : This file

History:
- 09/02/2022: Created original submodule
- ??/??/2022: Bug fix caused by implicit casting to integer values
- 02/03/2023: Fix license issue

Notice:
The code under this submodule is licensed under Apache License Version 2.0. All other code that is not third-party is licensed under GPLv3. Please read the LICENSE.txt located in this submodule for more information about the use of the modified and the original source code.