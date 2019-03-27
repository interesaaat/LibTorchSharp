#pragma once

#include "TH/THGeneral.h"
#include "torch/torch.h"

// Returns whether the grad is enabled or not.
TH_API bool THSAutograd_isGradEnabled();

// Enables / disables grad.
TH_API void THSAutograd_setGrad(bool enabled);
