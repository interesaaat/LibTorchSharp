#include "Utils.h"

#include <cstring>
#include <fstream>

extern thread_local char * torch_last_err = nullptr;

const char * make_sharable_string(const std::string str)
{
    size_t size = sizeof(str);
    char* result = new char[size];
    strncpy(result, str.c_str(), size);
    result[size - 1] = '\0';
    return result;
}

const char * get_and_reset_last_err() 
{
    char *tmp = torch_last_err;
    torch_last_err = nullptr;
    return tmp;
}
