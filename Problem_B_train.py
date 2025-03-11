# Helpful functions used through the entire notebook
import torch
import torch.nn as nn
from transformers import set_seed
import time
import inspect
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
import sys
import traceback
import functools
major_version, minor_version = torch.cuda.get_device_capability()
HAS_BFLOAT16 = (major_version >= 8)
from inspect import currentframe as _C, getframeinfo
_F = lambda c: getframeinfo(c).lineno # Gets line number
WARN = lambda x: print(f"\033[31m{x}\033[0m") # Red colored warnings

# https://stackoverflow.com/questions/18425225/getting-the-name-of-a-variable-as-a-string
def NAME(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    names = [var_name for var_name, var_val in callers_local_vars if var_val is var]
    return names[0] if len(names) != 0 else ""

def assert_same(x, y, line, dtype):
    assert(x.dtype == dtype)
    try: torch.testing.assert_close(x, y, check_stride = True)
    except Exception as error:
        raise RuntimeError(
            f"Failed allclose at line [{line}]: {NAME(x)}, {NAME(y)}\n{str(error)}"
        )

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,"\
    "roundup_power2_divisions:[32:256,64:128,256:64,>:32]"

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType

# HELPFUL functions to undo Unsloth patches:
import sys

def remove_patched_module(package_name):
    modules_to_delete = [
        name for name in sys.modules
        if name == package_name or name.startswith(package_name + ".")
    ]
    for name in modules_to_delete: del sys.modules[name]

remove_patched_module("trl")
remove_patched_module("transformers")
remove_patched_module("peft")
remove_patched_module("bitsandbytes")

# Check available GPUs
n_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {n_gpus}")


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

        # Create and wrap model with FSDP ==========
        max_seq_length = 2048
        torch.set_default_dtype(torch.float16)
        model_name = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"  # use smaller model
        # model_name = "unsloth/meta-Llama-3.1-8B-Instruct-bnb-4bit"
        dtype = torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            attn_implementation="sdpa",
            quantization_config=bnb_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.padding_side = "right"

        lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        # Get LoRA and setup model
        model = get_peft_model(model, lora_config)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if ".lora_A." in name or ".lora_B." in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

        # Get dataset
        from datasets import load_dataset
        from trl import SFTTrainer, SFTConfig
        url = "https://huggingface.co/datasets/laion/OIG/resolve/main/unified_chip2.jsonl"
        dataset = load_dataset("json", data_files={"train": url}, split="train[:10%]")

        # Auto-wrap only trainable (non-quantized) parameters
        # wrap_policy = lambda aa: lambda_auto_wrap_policy(
        #     aa,
        #     True,
        #     0,
        #     lambda m: any(p.requires_grad for p in m.parameters())
        # )
        # try simplifying to
        # my_auto_wrap_policy = functools.partial(
        #     size_based_auto_wrap_policy, min_num_params=100
        # )
        def auto_wrap_policy(module, recurse, nonwrapped_numel):
            # return False
            should_wrap = all(p.requires_grad for p in module.parameters())
            print(module, should_wrap, 'recurse', recurse)
            return all(p.requires_grad for p in module.parameters())

        print(model.get_input_embeddings().weight.dtype)
        fsdp_model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=rank,
            sync_module_states=True, # TODO understand
            use_orig_params=True, # TODO understand
            mixed_precision=MixedPrecision(param_dtype=torch.float16),
            auto_wrap_policy=auto_wrap_policy,

            # FIXME
            # mixed_precision=MixedPrecision(
            #     param_dtype=torch.bfloat16,  # Parameters in bfloat16 for memory savings
            #     reduce_dtype=torch.float32,  # Gradient reductions in float32 for stability
            #     buffer_dtype=torch.float16  # Buffers (e.g., batch norm) in float16
            # )
        )

        trainer = SFTTrainer(
            model=fsdp_model,
            train_dataset=dataset,
            processing_class=tokenizer,
            args=SFTConfig(
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                warmup_steps=1,
                max_steps=10,
                logging_steps=1,
                output_dir="outputs",
                seed=3407,
                max_seq_length=max_seq_length,
                fp16=fsdp_model.get_input_embeddings().weight.dtype == torch.float16,
                bf16=fsdp_model.get_input_embeddings().weight.dtype == torch.bfloat16,
                report_to="none",  # For W&B
                dataset_num_proc=4,
            ),
        )
        trainer.train()

        # Create and wrap model with FSDP ==========

        # Memory usage for confirmation
        mem = torch.cuda.memory_allocated(rank) / 1024**2
        print(f"Rank {rank}: Loss = ???, GPU memory = {mem:.2f} MB", file=sys.stderr)

        # Store result
        result_queue.put((rank, 123, mem))
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
