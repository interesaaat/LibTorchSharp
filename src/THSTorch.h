#pragma once

#include "Utils.h"

// API.

// Sets manually the seed.
THS_API void THSTorch_seed(const int64_t seed);

// Sets manually the seed.
THS_API bool THSTorch_isCudaAvailable();