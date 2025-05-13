# free_mem.py
import torch
import gc

# These are safe even if no variables are defined
gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

print("✅ CUDA memory cleanup completed.")

