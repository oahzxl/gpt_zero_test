import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer, GPT2Tokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration, GPT2LMHeadModel, GPT2Config
import functools
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers.models.t5.modeling_t5 import T5Block
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from dataset.fsdp_webtext import FSDPWebtextDataset

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
 checkpoint_wrapper,
 CheckpointImpl)

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from functools import partial
from torch.utils.data import DataLoader
from pathlib import Path
from summarization_dataset import *
from transformers.models.t5.modeling_t5 import T5Block
from typing import Type
import time
import tqdm
from datetime import datetime

def setup():
    # initialize the process group
    dist.init_process_group("nccl")

def cleanup():
    dist.destroy_process_group()
    
def setup_model(model_name):
    model = GPT2LMHeadModel(GPT2Config())
    tokenizer =  GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def get_date_of_run():
    """create date and time for file save uniqueness
    example: 2022-05-07-08:31:12_PM'
    """
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    print(f"--> current date and time of run = {date_of_run}")
    return date_of_run

def format_metrics_to_gb(item):
    """quick function to format numbers to gigabyte and round to 4 digit precision"""
    metric_num = item / (1024 ** 3)
    metric_num = round(metric_num, ndigits=4)
    return metric_num

def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
    model.train()
    local_rank = int(os.environ['LOCAL_RANK'])
    fsdp_loss = torch.zeros(2).to(local_rank)

    if sampler:
        sampler.set_epoch(epoch)
    if rank==0:
        inner_pbar = tqdm.tqdm(
            range(len(train_loader)), colour="blue", desc="r0 Training Epoch"
        )
    for batch in train_loader:
        for key in batch.keys():
            batch[key] = batch[key].to(local_rank)
        optimizer.zero_grad()
        output = model(input_ids=batch["source_ids"],attention_mask=batch["source_mask"], labels=batch["target_ids"])
        loss = output.loss
        loss.backward()
        optimizer.step()
        fsdp_loss[0] += loss.item()
        fsdp_loss[1] += len(batch)
        if rank==0:
            inner_pbar.update(1)

    dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
    train_accuracy = fsdp_loss[0] / fsdp_loss[1]


    if rank == 0:
        inner_pbar.close()
        print(
                f"Train Epoch: \t{epoch}, Loss: \t{train_accuracy:.4f}"
            )
    return train_accuracy

def validation(model, rank, world_size, val_loader):
    model.eval()
    correct = 0
    local_rank = int(os.environ['LOCAL_RANK'])
    fsdp_loss = torch.zeros(3).to(local_rank)
    if rank == 0:
        inner_pbar = tqdm.tqdm(
            range(len(val_loader)), colour="green", desc="Validation Epoch"
        )
    with torch.no_grad():
        for batch in val_loader:
            for key in batch.keys():
                batch[key] = batch[key].to(local_rank)
            output = model(input_ids=batch["source_ids"],attention_mask=batch["source_mask"],labels=batch["target_ids"])
            fsdp_loss[0] += output["loss"].item()  # sum up batch loss
            fsdp_loss[1] += len(batch)

            if rank==0:
                inner_pbar.update(1)

    dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
    val_loss = fsdp_loss[0] / fsdp_loss[1]
    if rank == 0:
        inner_pbar.close()
        print(f"Validation Loss: {val_loss:.4f}")
    return val_loss

def fsdp_main(args):

    model, tokenizer = setup_model("t5-base")

    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])


    dataset = load_dataset('wikihow', 'all', data_dir='data/')
    print(dataset.keys())
    print("Size of train dataset: ", dataset['train'].shape)
    print("Size of Validation dataset: ", dataset['validation'].shape)


    #wikihow(tokenizer, type_path, num_samples, input_length, output_length, print_text=False)
    # train_dataset = wikihow(tokenizer, 'train', 1500, 512, 512, False)
    # val_dataset = wikihow(tokenizer, 'validation', 300, 512, 512, False)
    train_dataset = FSDPWebtextDataset(os.environ['DATA'], seq_len=512)
    val_dataset = FSDPWebtextDataset(os.environ['DATA'], seq_len=512)
    
    sampler1 = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
    sampler2 = DistributedSampler(val_dataset, rank=rank, num_replicas=world_size)

    setup()


    train_kwargs = {'batch_size': args.batch_size, 'sampler': sampler1}
    test_kwargs = {'batch_size': args.test_batch_size, 'sampler': sampler2}
    cuda_kwargs = {'num_workers': 2,
                    'pin_memory': True,
                    'shuffle': False}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, **test_kwargs)

    t5_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            GPT2Block,
        },
    )
    torch.cuda.set_device(local_rank)


    #init_start_event = torch.cuda.Event(enable_timing=True)
    #init_end_event = torch.cuda.Event(enable_timing=True)

    #init_start_event.record()
    bfSixteen = MixedPrecision(
        param_dtype=torch.bfloat16,
        # Gradient communication precision.
        reduce_dtype=torch.bfloat16,
        # Buffer precision.
        buffer_dtype=torch.bfloat16,
    )  
    bf16_ready = (
    torch.version.cuda
    and torch.cuda.is_bf16_supported()
    and dist.is_nccl_available()
    )

    if bf16_ready:
        mp_policy = bfSixteen
    else:
        mp_policy = None # defaults to fp32

    # model is on CPU before input to FSDP
    model = FSDP(model,
        auto_wrap_policy=t5_auto_wrap_policy,
        mixed_precision=mp_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=torch.cuda.current_device()
    )
    parameter_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print("parameter_memory: %.4f" % parameter_memory)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    best_val_loss = float("inf")
    curr_val_loss = float("inf")
    file_save_name = "T5-model-"

    if rank == 0:
        time_of_run = get_date_of_run()
        dur = []
        train_acc_tracking = []
        val_acc_tracking = []
        training_start_time = time.time()

    if rank == 0 and args.track_memory:
        mem_alloc_tracker = []
        mem_reserved_tracker = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_accuracy = train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=sampler1)
        # if args.run_validation:
        #     curr_val_loss = validation(model, rank, world_size, val_loader)
        scheduler.step()

        if rank == 0:

            print(f"--> epoch {epoch} completed...entering save and stats zone")

            dur.append(time.time() - t0)
            train_acc_tracking.append(train_accuracy.item())

            if args.run_validation:
                val_acc_tracking.append(curr_val_loss.item())

            if args.track_memory:
                mem_alloc_tracker.append(
                    format_metrics_to_gb(torch.cuda.memory_allocated())
                )
                mem_reserved_tracker.append(
                    format_metrics_to_gb(torch.cuda.memory_reserved())
                )
            print(f"completed save and stats zone...")

        if args.save_model and curr_val_loss < best_val_loss:

            # save
            if rank == 0:
                print(f"--> entering save model state")

            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(
                model, StateDictType.FULL_STATE_DICT, save_policy
            ):
                cpu_state = model.state_dict()
            #print(f"saving process: rank {rank}  done w state_dict")


            if rank == 0:
                print(f"--> saving model ...")
                currEpoch = (
                    "-" + str(epoch) + "-" + str(round(curr_val_loss.item(), 4)) + ".pt"
                )
                print(f"--> attempting to save model prefix {currEpoch}")
                save_name = file_save_name + "-" + time_of_run + "-" + currEpoch
                print(f"--> saving as model name {save_name}")

                torch.save(cpu_state, save_name)

        if curr_val_loss < best_val_loss:

            best_val_loss = curr_val_loss
            if rank==0:
                print(f"-->>>> New Val Loss Record: {best_val_loss}")

    current_memory = torch.cuda.memory_allocated() / (1024 ** 3)
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
    grads_and_optimizer_states_memory = current_memory - parameter_memory
    activation_memory = peak_memory - current_memory
    print("current_memory: %.4f" % current_memory)
    print("grads_and_optimizer_states_memory: %.4f" % grads_and_optimizer_states_memory)
    print("activation_memory: %.4f" % activation_memory)

    dist.barrier()
    cleanup()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch T5 FSDP Example')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=4, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 3)')
    parser.add_argument('--lr', type=float, default=.002, metavar='LR',
                        help='learning rate (default: .002)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--track_memory', action='store_false', default=True,
                        help='track the gpu memory')
    parser.add_argument('--run_validation', action='store_false', default=True,
                        help='running the validation')
    parser.add_argument('--save-model', action='store_false', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    fsdp_main(args)