# Convert nanoGPT model to lit-gpt model
# Usage: python convertnanogpt.py --model_path /path/to/nanogpt/model --output_path /path/to/lit-gpt/model
# Example: python convertnanogpt.py --model_path /path/to/nanogpt/model --output_path /path/to/lit-gpt/model
# Note: This script is for converting nanoGPT model to lit-gpt model
import os
import tiktoken

import lightning as L
import torch

from lit_gpt import LitGPT
from generate.base import next_token
from lit_gpt import GPT, Config, Tokenizer
from lit_gpt.utils import check_valid_checkpoint_dir, get_default_supported_precision, load_checkpoint
from lit_gpt.utils import get_default_supported_precision, load_checkpoint

from model import GPTConfig, GPT

def load_model(model_name='', block_size = 1024, n_layer = 6, n_head = 8, n_embd = 256, models_dir = '/home/peter/GitHub/nanoGPT/models'):
    # block_size = 32 # 1024
    # n_layer = 12 # 12
    # n_head = 12 # 12
    # n_embd = 144 # 768

    if model_name == '':
        model_name = 'GPT_B'+str(block_size)+'_L'+str(n_layer)+'_H'+str(n_head)+'_E'+str(n_embd)
    
    step_text = ''
    model_name = model_name +'.pt'
    ckpt_path = os.path.join(models_dir, model_name)
    seed = 1337
    device = 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
    compile = False # use PyTorch 2.0 to compile the model to be faster
    enc = tiktoken.get_encoding("gpt2")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast

    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            #step_text = step_text + ' ' + k

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model) # requires PyTorch 2.0 (optional)
    
    return model, enc





    fabric = L.Fabric(devices=1, precision=precision, plugins=plugins)

    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
    with fabric.init_module(empty_init=True):
        model = GPT(config)
        # enable the kv cache
        model.set_kv_cache(batch_size=1)
    load_checkpoint(fabric, model, checkpoint_path)
    model.eval()


class GPT(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config

        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=config.lm_head_bias)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        self.max_seq_length = self.config.block_size
        self.mask_cache: Optional[torch.Tensor] = None

    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value: int) -> None:
        """
        When doing inference, the sequences used might be shorter than the model's context length.
        This allows setting a smaller number to avoid allocating unused memory
        """
        if value > self.config.block_size:
            raise ValueError(f"Cannot attend to {value}, block size is only {self.config.block_size}")
        self._max_seq_length = value
        if not hasattr(self, "cos"):
            # first call
            cos, sin = self.rope_cache()
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
        # override
        elif value != self.cos.size(0):
            self.cos, self.sin = self.rope_cache(device=self.cos.device)
        # the mask and kv cache size will get updated on `set_kv_cache`. we cannot update it here because we don't know
        # if the kv cache is expected

    def reset_parameters(self) -> None:
        # Trigger resetting the rope-cache
        self.cos, self.sin = self.rope_cache()

    def _init_weights(self, module: nn.Module) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
