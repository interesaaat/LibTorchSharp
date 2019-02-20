#include "utils.h"
#include <fstream>

// Uitlity method used to built sharable strings.
const char * makeSharableString(std::string str)
{
    size_t size = sizeof(str);
    char* result = new char[size];
    strncpy(result, str.c_str(), size);
    result[size - 1] = '\0';
    return result;
}

std::ofstream GetLog(std::string filename)
{
    std::ofstream logger;
    logger.open(filename);
    return logger;
}