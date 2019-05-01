#pragma once

#include "Utils.h"

// API.

// Sets manually the seed.
THS_API void THSTorch_seed(const int64_t seed);

// Sets manually the seed.
THS_API int THSTorch_isCudaAvailable();

// Returns the latest error. This is thread-local.
THS_API const char * THSTorch_get_and_reset_last_err();

// Returns a Scalar object from a char value.
THS_API Scalar THSTorch_btos(char value);

// Returns a Scalar object from a short value.
THS_API Scalar THSTorch_stos(short value);

// Returns a Scalar object from an int value.
THS_API Scalar THSTorch_itos(int value);

// Returns a Scalar object from a long value.
THS_API Scalar THSTorch_ltos(long value);

// Returns a Scalar object from a float value.
THS_API Scalar THSTorch_ftos(float value);

// Returns a Scalar object from a double value.
THS_API Scalar THSTorch_dtos(double value);

// Dispose the scalar.
THS_API void THSThorch_dispose_scalar(Scalar scalar);
