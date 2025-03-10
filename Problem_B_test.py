import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy
import sys
import traceback

# Check available GPUs
n_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {n_gpus}")
# assert n_gpus > 1, "This demo requires multiple GPUs"

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 10)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

# Training function with error handling
def train_fn(rank, world_size, result_queue):
    try:
        # Initialize distributed environment
        print(f"Rank {rank}: Initializing process group", file=sys.stderr)
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://127.0.0.1:29501",
            world_size=world_size,
            rank=rank
        )
        torch.cuda.set_device(rank)
        print(f"Rank {rank}: Process group initialized, using GPU {rank}", file=sys.stderr)

        # Create and wrap model with FSDP
        model = SimpleModel().cuda(rank)
        fsdp_model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=rank,
            sync_module_states=True,
            use_orig_params=True,
        )

        # Optimizer and data
        optimizer = torch.optim.Adam(fsdp_model.parameters(), lr=0.001)
        inputs = torch.randn(4, 1024).cuda(rank)
        targets = torch.randint(0, 10, (4,)).cuda(rank)

        # Training step
        fsdp_model.train()
        optimizer.zero_grad()
        outputs = fsdp_model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        loss.backward()
        optimizer.step()

        # Memory usage for confirmation
        mem = torch.cuda.memory_allocated(rank) / 1024**2
        print(f"Rank {rank}: Loss = {loss.item()}, GPU memory = {mem:.2f} MB", file=sys.stderr)

        # Store result
        result_queue.put((rank, loss.item(), mem))
        dist.destroy_process_group()

    except Exception as e:
        print(f"Rank {rank}: Error - {str(e)}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        result_queue.put((rank, None, None))  # Signal failure
        if dist.is_initialized():
            dist.destroy_process_group()

# Main execution
def run_demo():
    world_size = n_gpus
    result_queue = mp.Queue()

    # Spawn processes
    processes = []
    for rank in range(world_size):
        print('starting process..')
        p = mp.Process(target=train_fn, args=(rank, world_size, result_queue))
        p.start()
        processes.append(p)

    # Wait for processes and collect results
    results = []
    for _ in range(world_size):
        results.append(result_queue.get())  # Block until result is available

    for p in processes:
        p.join()

    # Display results
    for rank, loss, mem in sorted(results):
        if loss is not None:
            print(f"Rank {rank}: Loss = {loss}, GPU memory = {mem:.2f} MB")
        else:
            print(f"Rank {rank}: Failed to complete")

# Run the demo
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    run_demo()
