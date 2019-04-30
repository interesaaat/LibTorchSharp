#pragma once

#include "Utils.h"

// API.

// Sets manually the seed.
THS_API void THSTorch_seed(const int64_t seed);

// Sets manually the seed.
THS_API int THSTorch_isCudaAvailable();

// Returns the latest error. This is thread-local.
THS_API const char * THSTorch_get_and_reset_last_err();
