#ifndef CC_UTILS_H
#define CC_UTILS_H

// std
#include <vector>
#include <iostream>
#include <string>

std::string get_execution_type(int execution_type)
{
    if (execution_type == 0)
        return "pool";
    else
        return "bulk-pool";
}

#endif