import torch
import fused_stack_add

from torch.profiler import profile, ProfilerActivity
import torch


tensor_list = []
for i in range(20):
    tensor_list.append(torch.randn((10, 30, 40)).cuda()); # B 10 V 20 L 30 D 40

other = torch.randn((1, 20, 40), dtype=torch.float).cuda() # 1 V20 D40


# # ground truth
golden_cat = torch.stack(tensor_list, dim=1);
other_ = other.unsqueeze(-2)
golden_result = golden_cat + other_;
# print(golden_result)

# result by fused op
fused_result = fused_stack_add.forward(tensor_list, other, 1);
# print(fused_result)
print(fused_result == golden_result)
