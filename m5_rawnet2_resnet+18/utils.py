
def normalize_tensor(tensor):
    tensor = tensor.float()
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    normalized_tensor = normalized_tensor * 2 - 1
    return normalized_tensor
