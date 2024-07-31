import torch
from torch.profiler import profile, ProfilerActivity

with profile(
	activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
	on_trace_ready=torch.profiler.tensorboard_trace_handler("./torch_stack_add"),
	profile_memory=True,
	with_stack=True,
	record_shapes=True,
    use_cuda=True) as prof:
    tensor_list = []
    for i in range(10):
        tensor_list.append(torch.randn(48, 512, 1024).cuda());

    other = torch.randn((1, 10, 1024), dtype=torch.float).cuda()
    golden_cat = torch.stack(tensor_list, dim=1);
    golden_cat = golden_cat + other.unsqueeze(2)