#pragma once

#include <string>

#include "TH/THGeneral.h"

#define THS_API TH_API

// Utility method used to built sharable strings.
const char * makeSharableString(std::string str);

// Utility method used for debugging through logging.
std::ofstream GetLog(std::string filename);
