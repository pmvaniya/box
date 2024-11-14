import os
import sys
import time
import requests
import tarfile
import json
import math
import tiktoken
import inspect
import torch
import numpy as np
import torch.nn as nn

from pathlib import Path
from shutil import rmtree
from dataclasses import dataclass
from torch.nn import functional as F
from transformers import GPT2LMHeadModel

# the necessary functions will be called if set to True
DOWNLOAD_OWT = False
EXTRACT_OWT = False
CONVERT_OWT = False
TRAIN_LLM = False

# some useful constants
CURR_DIR = Path(__file__).parent
DATA_DIR = CURR_DIR / "data"
OWT_DIR = DATA_DIR / "OpenWebText"
EXTRACT_DIR = DATA_DIR / "extracted"
CONVERT_DIR = DATA_DIR / "converted"
TOKEN_ID_FILE = DATA_DIR / "token_ids.json"

# some other useful variables
stoi = json.load(open(TOKEN_ID_FILE, "r"))
itos = {i: s for s, i in stoi.items()}
max_token_length = max([len(s) for s, i in stoi.items()])
shard_size = 25_000_000  # 25M tokens per shard, roughly 48MB

# https://raw.githubusercontent.com/karpathy/char-rnn/refs/heads/master/data/tinyshakespeare/input.txt


# download OpenWebText
def download_owt(des_dir) -> None:
    if os.path.exists(des_dir):
        rmtree(des_dir)

    os.mkdir(des_dir)

    total_subset_files = 21
    counter = 1

    links = [
        f"https://huggingface.co/datasets/Skylion007/openwebtext/resolve/main/subsets/urlsf_subset{i:02}.tar?download=true"
        for i in range(total_subset_files)
    ]

    for link in links:
        name = link.split("/")[-1].split("?")[0]
        download_path = des_dir / name
        print(f"Downloading {name} (file {counter} of {total_subset_files})")

        try:
            response = requests.get(link)
            open(download_path, "wb").write(response.content)
        except:
            print(f"Internet?")
            exit()


# extract the downloaded OpenWebText
def extract_owt(src_dir, des_dir) -> None:
    tars = sorted(os.listdir(src_dir))
    subtar_dir = des_dir / "openwebtext"
    counter = 0

    if os.path.exists(des_dir):
        rmtree(des_dir)

    for tar in tars:
        print("extracting", tar)
        tar_path = OWT_DIR / tar
        extraction_path = des_dir / f"subset{counter:02}"

        with tarfile.open(tar_path, "r") as archive:
            archive.extractall(path=des_dir, filter="tar")
            subtars = os.listdir(subtar_dir)

            for subtar in subtars:
                subtar_path = subtar_dir / subtar

                with tarfile.open(subtar_path, "r") as subarchive:
                    subarchive.extractall(path=extraction_path, filter="tar")

            rmtree(subtar_dir)

        counter += 1


# convert extracted textfiles to numpy arrays
def convert_owt(src_dir, des_dir):
    if os.path.exists(des_dir):
        rmtree(des_dir)

    os.mkdir(des_dir)

    print("starting conversion")
    textfile_dirs = sorted(os.listdir(src_dir))
    np_array = np.zeros(shard_size, dtype=np.uint16)
    token_counter = 0
    shard_counter = 0

    for textfile_dir in textfile_dirs:
        textfile_dir_path = src_dir / textfile_dir
        textfiles = os.listdir(textfile_dir_path)

        for textfile in textfiles:
            textfile_path = textfile_dir_path / textfile
            text = open(textfile_path, "r").read()
            encoded = encode_text(text)
            encoded_length = len(encoded)

            if token_counter + encoded_length < shard_size:
                np_array[token_counter : token_counter + encoded_length] = encoded
                token_counter += encoded_length

            else:
                np_array[token_counter:shard_size] = encoded[
                    : (shard_size - token_counter)
                ]
                np.save(des_dir / f"array{shard_counter:04}.npy", np_array)
                print(f"saved array{shard_counter:04}.npy")

                token_counter = 0
                shard_counter += 1
                np_array = np.zeros(shard_size, dtype=np.uint16)


def encode_text(text):
    text_length = len(text)
    encoded_text = []
    encoded_text.append(stoi["<|S|>"])
    i = 0

    # Greedy tokenization
    while i < text_length:
        matched = False

        # Ensure that we only check lengths up to max_token_length
        max_len = min(max_token_length, text_length - i)

        for length in range(max_len, 0, -1):
            token = text[i : i + length]

            if token in stoi:
                encoded_text.append(stoi[token])
                i += length
                matched = True
                break

        if not matched:
            encoded_text.append(stoi["<|OoV|>"])
            i += 1

    encoded_text.append(stoi["<|E|>"])

    return encoded_text


def decode_tokens(tokens):
    return "".join([itos[id] for id in tokens])


def train_llm(src_dir):
    # print("TRAIN LLM")
    pass


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, n_embd

        qkv = self.c_attn(x)  # query, key, values
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # att = F.softmax(att, dim=-1)
        # y = att @ v
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)

        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)

        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    # vocab_size: int = 50257
    vocab_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params} parameters"
        )

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in device

        print(f"using fused AdamW: {use_fused}")

        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )

        return optimizer

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02

            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5

            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()

        assert T <= self.config.block_size, f"cannot forward sequence length {T}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}

        print(f"loading weights from pretrained gpt: {model_type}")

        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]

        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024

        config = GPTConfig(**config_args)
        model = GPT(config)

        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]

        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape

                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())

            else:
                assert sd_hf[k].shape == sd[k].shape

                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open("input.txt", "r") as f:
            text = f.read()

        enc = tiktoken.get_encoding("gpt2")
        # tokens = enc.encode(text)
        tokens = encode_text(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens)//(B*T)} batches")

        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B * T

        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0

        return x, y


if os.path.exists(DATA_DIR) == False:
    os.mkdir(DATA_DIR)

if DOWNLOAD_OWT == True:
    download_owt(des_dir=OWT_DIR)

if EXTRACT_OWT == True:
    extract_owt(src_dir=OWT_DIR, des_dir=EXTRACT_DIR)

if CONVERT_OWT == True:
    convert_owt(src_dir=EXTRACT_DIR, des_dir=CONVERT_DIR)

if TRAIN_LLM == True:
    train_llm(src_dir=CONVERT_DIR)

sys.exit(0)

device = "cpu"

if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"

device = "cpu"  # override

print(f"Using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

B = 4  # micro batch size
T = 128  # sequence length
total_batch_size = 524288  # 2**19, ~0.5M, in number of tokens
assert total_batch_size % (B * T) == 0, "total batch size is not divisible by B*T"
grad_accum_steps = total_batch_size // (B * T)
print(f"total desired batch size: {total_batch_size}")
print(f"calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T)

torch.set_float32_matmul_precision("high")

model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
model = torch.compile(model)

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50


def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps

    if it > max_steps:
        return min_lr

    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_lr + coeff * (max_lr - min_lr)


optimizer = model.configure_optimizers(
    weight_decay=0.1, learning_rate=6e-4, device=device
)

for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0

    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps) / dt
    print(
        f"step {step+1:4d} | loss: {loss_accum.item():.6f} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
    )

# exit
sys.exit(0)

num_return_sequences = 1
max_length = 32

enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to(device)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)
        x = torch.cat((x, xcol), dim=1)

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
