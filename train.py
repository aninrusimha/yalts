import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np

import glob, os, sys
import tqdm

from yalts.arguments import get_training_arguments
from yalts.utils import pretty_tokens_str, get_num_params, print_rank_0
from yalts.model import Transformer
from yalts.data import MemMapBinaryDataset


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
    assert args.global_batch_tokens % args.micro_batch_tokens * world_size == 0

    # Initialize the model, compile it, and wrap it in DDP
    # TODO: add mixed precision support
    model = Transformer(args).to(device, dtype=torch.bfloat16)

    # model = torch.compile(model)
    model = DDP(model)

    # TODO: add loss, optimizer, and LR schedule

    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        fused=True,
    )

    # for the scheduler, add warmup and cosine or sqrt LR decay
    global_steps = args.num_tokens // args.global_batch_tokens
    # we usually cosine to be max(.1 *lr, cos)
    cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, global_steps)
    warmup_scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1, step / (args.warmup_ratio * global_steps))
    )

    # TODO: create the Dataset

    files = None

    train_dataset = MemMapBinaryDataset(
        args.dataset_path + "/train.bin",
    )
    val_dataset = MemMapBinaryDataset(args.dataset_path + "/val.bin")

    train_sampler = DistributedSampler(train_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.micro_batch_seqs,
        num_workers=2,
        pin_memory=True,
        shuffle=False,
    )

    model.train()

    # TODO: calculate flops or tokens or seqs and create a tqdm bar

    cur_tokens = 0
    flops_per_token = (
        get_num_params(model)
        - get_num_params(model.module.embedding)
        + (args.n_q_heads * args.max_seq_len**2 / 2 * args.d_head / args.max_seq_len)
        * 2
        * args.n_layers
    ) * 6
    tqdm_bar = tqdm.tqdm(
        total=(args.num_tokens * flops_per_token).item(), unit="FLOPS", unit_scale=True
    )
    for inputs in train_loader:
        if cur_tokens >= args.num_tokens:
            break
        # why is the line below correct?
        cur_tokens += args.micro_batch_tokens * world_size
        inputs = inputs.to(device)
        labels = inputs.clone()[..., 1:].reshape(-1).contiguous()

        # TODO: add gradient accumulation
        # TODO: add mixed precision support
        if cur_tokens % args.global_batch_tokens == 0:
            inputs = inputs.to(device)
            labels = inputs.clone()[..., 1:].reshape(-1).contiguous()
            outputs = model(inputs)[..., :-1, :].reshape(-1, args.vocab_size)
            loss = loss_fn(outputs, labels)
            loss.backward()
            torch.cuda.synchronize()
            optimizer.step()
            cos_scheduler.step()
            warmup_scheduler.step()
            optimizer.zero_grad()
            tqdm_bar.update((flops_per_token * args.global_batch_tokens).item())
        else:
            with model.no_sync():
                inputs = inputs.to(device)
                labels = inputs.clone()[..., 1:].reshape(-1).contiguous()
                outputs = model(inputs)[..., :-1, :].reshape(-1, args.vocab_size)
                loss = loss_fn(outputs, labels)
                loss.backward()

        # TODO: add logging and checkpointing

    torch.distributed.destroy_process_group()

    # TODO: save the model


if __name__ == "__main__":
    train()
