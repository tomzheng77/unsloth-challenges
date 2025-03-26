# Problem A

The main idea behind the implementation is to handle one absmax block (256 absmax values, for a total of 256 x 64 = 16384 values)
per triton kernel. This worked reasonably well compared to reduced kernel sizes (which slowed down performance during testing.) When
testing on a Tesla T4 it achieved a 1.166x speedup.

# Problem B

I used the `accelerate` library to spin up two subprocesses in the notebook.
I converted the Params4bit to float32 (as the storage_dtype) using a custom triton kernel
This allowed the LlamaAttention and LlamaMLP to be wrapped in one FSDP module each (as all their inner params are float32)
Therefore modules such as LlamaMLP can be torch compiled

https://www.kaggle.com/code/tomzheng77/problem-b-fsdp2/edit

# Problem C

I monkey patched the dequantize function to use what was developed for part A.
Then I monkey patched the 4bit forward function to avoid referencing Params4bit.data, which confuses it

# Problem E

I followed the ideas discussed in the puzzles notebook, reducing VRAM by 50%
