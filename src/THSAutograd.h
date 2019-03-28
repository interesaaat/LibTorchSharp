#pragma once

#include "Utils.h"

// Returns whether the grad is enabled or not.
THS_API bool THSAutograd_isGradEnabled();

// Enables / disables grad.
THS_API void THSAutograd_setGrad(bool enabled);
