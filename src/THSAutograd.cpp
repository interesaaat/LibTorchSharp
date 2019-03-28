#include "THSAutograd.h"

#include "torch/torch.h"

bool THSAutograd_isGradEnabled()
{
    return torch::autograd::GradMode::is_enabled();
}

void THSAutograd_setGrad(bool enabled)
{
    torch::autograd::GradMode::set_enabled(enabled);
}