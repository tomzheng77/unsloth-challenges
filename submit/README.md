# Problem A

The main idea behind the implementation is to handle one absmax block (256 absmax values, for a total of 256 x 64 = 16384 values)
per triton kernel. This worked reasonably well compared to reduced kernel sizes (which slowed down performance during testing.) When
testing on a Tesla T4 it achieved a 1.166x speedup.

# Problem B

I used the `accelerate` library and the 

# Problem C

I made sure torch compile works using the following ideas

- Monkey patching ??? is required to avoid the CUDA calls
- Monkey patch forward is to avoid calling Params4bit.t

# Problem E

Following the idea in described in the original notebook