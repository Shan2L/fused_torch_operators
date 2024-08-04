import torch
import fused_stack_add_index

from torch.profiler import profile, ProfilerActivity
import torch


tensor_list = []
for i in range(20):
    tensor_list.append(torch.randn((10, 30, 40)).cuda().half()); # B 10 V 20 L 30 D 40

other = torch.randn((1, 20, 40), dtype=torch.float).cuda().half() # 1 V20 D40

index = torch.tensor([1,2,3,4,5]).cuda();
print(index.dtype)

# # ground truth
golden_cat = torch.stack(tensor_list, dim=1);
other_ = other.unsqueeze(-2)
golden_result = golden_cat + other_;
golden_result = golden_result[:, :, index, :]
print(golden_result[0])

# result by fused op
fused_result = fused_stack_add_index.forward(tensor_list, other, index, 1, 2);
print(torch.all(fused_result == golden_result))
