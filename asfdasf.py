import torch
print("Torch version:", torch.__version__)
print("Torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")