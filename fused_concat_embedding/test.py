import torch
import fused_cat_embedding


a = torch.randint(low=0, high=10, size=(1, 48, 512, 1024)).int().cuda();
b = torch.randint(low=0, high=10, size=(1, 48, 512, 1024)).int().cuda()
c = torch.randint(low=0, high=10, size=(1, 512, 1024)).int().cuda();
weight = torch.randn((11, 10), dtype=torch.float).cuda()

fused_out = fused_cat_embedding.fused_cat_embedding([a, b, c], weight, 0)

golden_cat = torch.cat([a, b, c], dim=0);
golden_result = torch.nn.functional.embedding(golden_cat, weight)

print(fused_out == golden_result)
