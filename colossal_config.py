import torch
from colossalai.nn.optimizer import HybridAdam
from torch.optim import Adam
from colossalai.zero.shard_utils import TensorShardStrategy
from titans.model.gpt.gpt import gpt2_15B as gpt
from colossalai.amp import AMP_TYPE

"""
gpt=large batch=16 gpu=4 checkpoint=false:
colossal: 1.16it/s, 305.91 Tflops
fsdp: 1.84s/it=0.54it/s, 142.40 Tflops

gpt=15B batch=4 gpu=4 checkpoint=true:
colossal: 5.61s/it, 157.24 Tflops
fsdp: 3.16s/it
"""

BATCH_SIZE = 4
NUM_EPOCHS = 60
SEQ_LEN = 512


zero = dict(
    model_config=dict(
        shard_strategy=TensorShardStrategy(),
        reuse_fp16_shard=True
    ),
    optimizer_config=dict()
)


optimizer = dict(
    type=Adam,
    lr=0.00015,
    weight_decay=1e-2,
)

model = dict(
    type=gpt,
    checkpoint=True,
    dtype=torch.half,
)
