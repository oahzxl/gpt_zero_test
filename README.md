# Run GPT With Colossal-AI and FSDP

```bash
# colossal
DATA=/data2/users/lczxl/gpt/small-gpt-dataset.json colossalai run --nproc_per_node=4 run_colossal.py --from_torch --config=colossal_config.py
# fsdp
DATA=/data2/users/lczxl/gpt/small-gpt-dataset.json torchrun --nnodes 1 --nproc_per_node 4 fsdp.py
```