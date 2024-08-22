import torch

mults = {"k": 1024, "m": 1024**2, "b": 1024**3}


def pretty_tokens_str(tokens):
    if tokens / mults["b"] >= 1:
        return f"{int(tokens / mults['b'])}B"
    if tokens / mults["m"] >= 1:
        return f"{int(tokens / mults['m'])}M"
    if tokens / mults["k"] >= 1:
        return f"{int(tokens / mults['k'])}K"


def get_num_params(model):
    tot = 0
    for p in model.parameters():
        if p.requires_grad:
            tot += torch.prod(torch.tensor(p.size()))
    return tot


def print_rank_0(*args, **kwargs):
    if torch.distributed.get_rank() == 0:
        print(*args, **kwargs)
