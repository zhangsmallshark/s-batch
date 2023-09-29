#!/bin/bash --login

export TORCH_EXTENSIONS_DIR=/home/czh5/.cache/polaris_torch_extensions

# module load conda/2023-01-10-unstable ; conda activate base
# source /home/czh5/genome/Megatron-LM/venvs/polaris/2023-01-10/bin/activate

python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 mnist-dist.py --nodes=1 --gpus=4

