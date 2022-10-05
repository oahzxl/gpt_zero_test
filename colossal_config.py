import torch
from colossalai.nn.optimizer import HybridAdam
from torch.optim import Adam
from colossalai.zero.shard_utils import TensorShardStrategy
from titans.model.gpt.gpt import gpt2_40B as gpt
from colossalai.amp import AMP_TYPE


BATCH_SIZE = 8
NUM_EPOCHS = 60
SEQ_LEN = 512

# offload
# 4gpu 40B batch 8:  52.9
# 4gpu 40B batch24: 147.7

# parallel = dict(
#     pipeline=24,
#     tensor=dict(mode='2d', size=4)
# )

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
