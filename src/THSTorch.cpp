#include "THSTorch.h"

#include "torch/torch.h"

void THSTorch_seed(const int64_t seed)
{
    torch::manual_seed(seed);
}

bool THSTorch_isCudaAvailable()
{
    bool result = torch::cuda::is_available();
    return result;
}
