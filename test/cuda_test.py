import torch

if torch.cuda.is_available():
    print("CUDA is available")
    print("Device Name:", torch.cuda.get_device_name(0))
    print("Total GPUs:", torch.cuda.device_count())
else:
    print("CUDA is NOT available")
