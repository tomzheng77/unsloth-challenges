import os
os.environ['UNSLOTH_IS_PRESENT'] = '1'
from unsloth_zoo.rl_replacements import UnslothEfficientGRPO
import torch

old_hidden_states = torch.randn(6, 241, 2048, dtype=torch.bfloat16, device="cuda", requires_grad=True)
new_hidden_states = old_hidden_states # don't deviate too much to blow out K-L divergence
lm_head = torch.randn(128256, 2048, dtype=torch.bfloat16, device="cuda", requires_grad=True)
completion_input_ids = torch.randint(0, 128256, (6, 240), dtype=torch.int64, device="cuda")

# filter out 128004
# completion_mask = torch.randint(0, 2, (6, 240), dtype=torch.int64, device="cuda")
completion_mask = torch.ones_like(completion_input_ids)
advantages = torch.randn(6, dtype=torch.float32, device="cuda")
# advantages = torch.zeros(6, dtype=torch.float32, device="cuda")
beta = 0.04
scaler = None
n_chunks = 6

result = UnslothEfficientGRPO.apply(
    new_hidden_states,
    old_hidden_states,
    lm_head,
    completion_input_ids,
    completion_mask,
    advantages,
    beta,
    scaler,
    n_chunks,
)

print(result)
