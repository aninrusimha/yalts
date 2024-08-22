import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np

import glob, os, sys
from tqdm import tqdm

from yalts.arguments import get_training_arguments
from yalts.utils import pretty_tokens_str, get_num_params, print_rank_0
from yalts.model import Transformer
from yalts.data import DummyDataset


def train():

    # DDP boilerplate
    torch.distributed.init_process_group("nccl")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    device_count = torch.cuda.device_count()
    device = torch.device(
        f"cuda:{rank % device_count}" if torch.cuda.is_available() else "cpu"
    )

    # get the training arguments
    args = get_training_arguments()

    # Initialize the model, compile it, and wrap it in DDP
    # TODO: add mixed precision support
    model = Transformer(args).to(device, dtype=torch.bfloat16)

    model = torch.compile(model)

    # TODO: add loss, optimizer, and LR schedule

    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
    )

    # for the scheduler, add warmup and cosine or sqrt LR decay
    global_steps = args.num_tokens // args.global_batch_tokens
    cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, global_steps)
    warmup_scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1, step / (args.warmup_ratio * global_steps))
    )

    # TODO: create the Dataset

    files = None

    dataset = DummyDataset()

    train_sampler = DistributedSampler(dataset, shuffle=True)

    train_loader = DataLoader(
        dataset,
        sampler=train_sampler,
        batch_size=args.micro_batch_seqs,
        num_workers=1,
        shuffle=False,
    )

    model.train()

    # TODO: calculate flops or tokens or seqs and create a tqdm bar

    for inputs in tqdm(train_loader):
        inputs = inputs.to(device)
        # why is the line below correct?
        labels = inputs.clone()[..., 1:].reshape(-1).contiguous()

        # TODO: add gradient accumulation
        # TODO: add mixed precision support
        outputs = model(inputs)[..., :-1, :].reshape(-1, args.vocab_size)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        cos_scheduler.step()
        warmup_scheduler.step()

        # TODO: add logging and checkpointing

    torch.distributed.destroy_process_group()

    # TODO: save the model


if __name__ == "__main__":
    train()
