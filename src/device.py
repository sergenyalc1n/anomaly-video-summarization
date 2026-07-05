import torch

def get_device():
    # Mac (MPS) code commented out for Windows
    # if torch.backends.mps.is_available():
    #     return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
