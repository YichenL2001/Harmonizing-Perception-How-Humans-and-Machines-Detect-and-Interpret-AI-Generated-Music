import time
import torch
import pandas as pd
from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis


def profile_model(model, input_tensor, display=False):
    flops = calculate_flops(model, input_tensor[0:1, ...])  # (1, n_mels, n_frames)
    acts = calculate_activations(model, input_tensor[0:1, ...])  # (1, n_mels, n_frames)
    params = calculate_params(model)
    speed = calculate_speed(model, input_tensor[0:1, ...])  # (1, n_mels, n_frames)
    memory = calculate_memory(model, input_tensor)  # (B, n_mels, n_frames)
    profile_data = {
        "Metric": [
            "FLOPs (G)",
            "Activations (M)",
            "Params (M)",
            "Memory (GB)",
            "Speed (A/S)",
        ],
        "Value": [flops, acts, params, memory, speed],
    }
    profile_df = pd.DataFrame(profile_data).set_index("Metric").T
    if display:
        print(profile_df.to_markdown(index=False, tablefmt="grid"))
    return profile_df


def calculate_speed(model, input_tensor, num_runs=100, warmup_runs=5):
    model.eval()

    if torch.cuda.is_available():
        # Warm-up iterations
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(input_tensor)

        # Create CUDA events for timing
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # Actual timing
        start.record()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(input_tensor)
        end.record()

        # Synchronize to wait for the events to be recorded
        torch.cuda.synchronize()

        # Calculate elapsed time
        elapsed_time = start.elapsed_time(end)  # in milliseconds
        latency = elapsed_time / num_runs / 1000.0  # convert to seconds
    else:
        # Warm-up iterations
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(input_tensor)

        # Actual timing
        start = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(input_tensor)
        end = time.time()

        # Calculate elapsed time
        latency = (end - start) / num_runs

    return 1.0 / latency


def calculate_flops(model, input_tensor):
    """Calculate FLOPs in GigaFLOPs.
    Models often reports MACs as FLOPs e.g. ConvNeXt, timm library
    Reference:
    1. https://github.com/huggingface/pytorch-image-models/blob/main/benchmark.py#L206
    2. https://github.com/facebookresearch/fvcore/issues/69
    """
    flops = FlopCountAnalysis(model, input_tensor).total()
    return flops / 1e9  # in GigaFLOPs


def calculate_activations(model, input_tensor):
    acts = ActivationCountAnalysis(model, input_tensor).total()
    return acts / 1e6  # in Millions


def calculate_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6  # in Millions


def calculate_memory(model, input_tensor):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device=None)
        start_memory = torch.cuda.max_memory_allocated(device=None)
        model.train()
        _ = model(input_tensor)
        end_memory = torch.cuda.max_memory_allocated(device=None)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device=None)
        memory = (end_memory - start_memory) / (1024**3)  # in GB
    else:
        memory = 0
    return memory
