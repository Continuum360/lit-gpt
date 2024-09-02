import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from pathlib import Path
import sys
from typing import Optional, Union

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import Config, Tokenizer
from lit_gpt.model import GPT, Block
from lit_gpt.utils import chunked_cross_entropy, estimate_flops, get_default_supported_precision, num_parameters

model_name = "pythia-31m" # pythia-14m # pythia-31m # pythia-70m
name = "openwebtext"
out_dir = Path("out") / model_name
data_dir = Path("data") / name
save_interval = 100
eval_interval = 200
eval_iters = 20
log_interval = 50

# Hyperparameters
learning_rate = 6e-4
batch_size = 12 #125
micro_batch_size = 6 # 5
gradient_accumulation_steps = batch_size // micro_batch_size
assert gradient_accumulation_steps > 0
max_iters = 600000  # num_epochs * (epoch_size // micro_batch_size) // devices
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 2000
lr_decay_iters = max_iters
min_lr = 6e-5

hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}
logger = CSVLogger("out", name, flush_logs_every_n_steps=log_interval)


def setup(rank: int, world_size: int, precision: Optional[str] = None, resume: Union[bool, Path] = False) -> None:
    init_process_group(backend="gloo", rank=rank, world_size=world_size)
    resume = bool(1)
    precision = precision or get_default_supported_precision(training=True)

    # Initialize model, optimizer, etc.
    model = GPT(Config())
    model = DDP(model)

    # Add your model training code here

    destroy_process_group()

def main():
    world_size = 4  # Number of CPU cores to use
    mp.spawn(setup, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()

    