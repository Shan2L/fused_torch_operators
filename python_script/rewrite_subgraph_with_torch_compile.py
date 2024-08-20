from torch._dynamo import register_backend
from torch.fx import GraphModule
import fused_stack_add_index
from typing import Sequence, List
import torch.nn as nn
import operator
import torch

class TorchModel(nn.Module):
    def __init__(self):
        super(TorchModel, self).__init__()
    
    def forward(self, tensor_list, other, index):
        golden_cat = torch.stack(tensor_list, dim=1);
        golden_result = golden_cat + other.unsqueeze(2)
        golden_result = golden_result[:, :, index, :]
        return golden_result

def pass_stack_add_index_fusion(gm):
    graph = gm.graph
    
    # 用来存储已经匹配到的节点
    nodes_to_remove = []
    print(graph)
    for node in graph.nodes:
        if node.op == "call_function" and node.target == torch.stack:
            stack_node = node
            print("find stack")
            for stack_node_use in stack_node.users.keys():
                if stack_node_use.op == "call_function" and stack_node_use.target == operator.add\
                and stack_node_use.args[1].op == "call_method" and stack_node_use.args[1].target == "unsqueeze":
                    print("find add")
                    add_node = stack_node_use
                    for add_node_use in add_node.users.keys():
                        if add_node_use.op == "call_function" and add_node_use.target == operator.getitem:
                            print("find getitem")

                            index_node = add_node_use

                            # # 创建新的融合算子节点
                            return_node = graph.find_nodes(op='output')[0]
                            with graph.inserting_before(return_node):
                                fused_node = graph.call_function(
                                    fused_stack_add_index.forward,  # 替换为你的自定义算子
                                    (stack_node.args[0],  # inputs for torch.stack
                                    stack_node_use.args[1].args[0],  # input for unsqueeze (before unsqueeze)
                                    index_node.args[1][2],  # index for getitem
                                    stack_node.kwargs['dim'],  # stack_dim
                                    2)  # index_dim (unsqueeze adds at dim=2)
                                )
                            # 将 getitem 的所有使用替换为新的融合节点
                            index_node.replace_all_uses_with(fused_node)

                            # 将已经匹配到的节点存储到 nodes_to_remove 中
                            nodes_to_remove.extend([ index_node, add_node, stack_node_use.args[1], stack_node])
    for node in nodes_to_remove:
        graph.erase_node(node)
    
    print(graph)

    return gm

@register_backend
def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    gm = pass_stack_add_index_fusion(gm)
    return gm

model = TorchModel()
model.eval()

tensor_list = []
for i in range(10):
    tensor_list.append(torch.randn((48, 512, 1024)).cuda()); 

weight = torch.randn((1, 10, 1024), dtype=torch.float).cuda()
index = torch.tensor([1, 2, 3 ,4, 5, 6, 7, 8, 9, 10]).cuda()

golden_result = model(tensor_list, weight, index)

compile_model = torch.compile(model, backend=my_compiler)
fused_result = compile_model(tensor_list, weight, index)

print(torch.all(fused_result == golden_result))