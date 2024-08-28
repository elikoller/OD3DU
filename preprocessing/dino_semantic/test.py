import torch

# Print PyTorch version
print("PyTorch version:", torch.__version__)

# Check if CUDA is available
print("CUDA available:", torch.cuda.is_available())

# Print the current CUDA version
if torch.cuda.is_available():
    print("Current CUDA version:", torch.version.cuda)

# Print the name of the current GPU device
if torch.cuda.is_available():
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))