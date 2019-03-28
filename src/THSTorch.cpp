#include "THSTorch.h"

#include "torch/torch.h"

void THSTorch_seed(const int64_t seed)
{
    torch::manual_seed(seed);
}

bool THSTorch_isCudaAvailable()
{
    return torch::cuda::is_available();
}
