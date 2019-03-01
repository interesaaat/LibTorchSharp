#pragma once

#include <string>

// Utility method used to built sharable strings.
const char * makeSharableString(std::string str);

std::ofstream GetLog(std::string filename);
