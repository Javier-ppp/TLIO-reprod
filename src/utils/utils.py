import torch

def make_recursive_func(func):
    def wrapper(args, device=None):
        if isinstance(args, list):
            return [wrapper(x, device) for x in args]
        elif isinstance(args, tuple):
            return tuple([wrapper(x, device) for x in args])
        elif isinstance(args, dict):
            return {k: wrapper(v, device) for k, v in args.items()}
        else:
            return func(args, device)

    return wrapper


@make_recursive_func
def to_device(args, device):
    if isinstance(args, torch.Tensor):
        if device.type == 'mps' and args.dtype == torch.float64:
            return args.to(device, dtype=torch.float32)
        return args.to(device)
    elif isinstance(args, str) or isinstance(args, float) or isinstance(args, int):
        return args
    else:
        raise NotImplementedError(f"to_device Not implemented for type {type(args)}")
