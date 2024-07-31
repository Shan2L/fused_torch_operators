## An implementation for fused stack-add operator.
### How to build?
```bash
    pip install torch
```
```bash
    python setup.py
```

#### 2. How to test?
```bash
    import torch
    import fused_stack_add

    result = fused_stack_add.forward([a, b, c, ...], x, 1)
```
`a, b, c...` is a list of tensor, who has the same shape [B, L, D], the number of its elemets is V,
     and x is a tensor with shape [1, V, D]

