import torch
from torch.profiler import profile, ProfilerActivity

with profile(
	activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
	on_trace_ready=torch.profiler.tensorboard_trace_handler("./torch_stack_add_index"),
	profile_memory=True,
	with_stack=True,
	record_shapes=True,
    use_cuda=True) as prof:
    tensor_list = []
    for i in range(10):
        tensor_list.append(torch.randn(48, 512, 1024).cuda());

    other = torch.randn((1, 10, 1024), dtype=torch.float).cuda()
    index = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8,9,10]).cuda()
    golden_cat = torch.stack(tensor_list, dim=1);
    golden_result = golden_cat + other.unsqueeze(2)

    golden_result = golden_result[:, :, index, :]
    print(golden_result.shape)