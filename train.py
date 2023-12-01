from itertools import repeat
from pathlib import Path

import pandas as pd
import torch
from sentencepiece import SentencePieceProcessor
from torch import nn
from torch.distributions import Categorical
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from model import TransformerModel
from text_preparation import Collator, TokenizedDataset


def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


def number_of_weights(nn):
    return sum(p.numel() for p in nn.parameters() if p.requires_grad)


@torch.no_grad()
def get_grad_norm(model, norm_type=2):
    parameters = model.parameters()
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]),
        norm_type,
    )
    return total_norm.item()


@torch.inference_mode()
def inference(model, sp_model, max_length=500, prompt="Once upon a time there was"):
    model.eval()
    device = model.device

    prompt = torch.tensor([[sp_model.bos_id()] + sp_model.encode(prompt)]).to(device)
    for _ in range(max_length):
        logits = model(prompt)[0, -1]
        new_token = Categorical(logits=logits).sample().unsqueeze(0).unsqueeze(0)
        if new_token.item() == sp_model.eos_id():
            break
        prompt = torch.cat([prompt, new_token], axis=1)
    return sp_model.decode(prompt.squeeze().tolist())


config = {
    "vocab_size": 5000,  # size of vocabulary
    "emsize": 384,  # embedding dimension
    "d_hid": 384,  # dimension of the feedforward network model in ``nn.TransformerEncoder``
    "nlayers": 4,  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
    "nhead": 16,  # number of heads in ``nn.MultiheadAttention``
    "dropout": 0.1,  # dropout probability
    "max_length": 512,
    "batch_size": 160,
    "lr": 5e-4,
    "weight_decay": 0.1,
    "epochs": 100,
    "len_epoch": 10000,
    "log_step": 100,
    "accumulation_steps": 2,
    "project": "boutique_lm",
    "name": "Medium model, 7.5kk parameters (fixed loss accumulation)",
    "save_period": 1,
}
config["sp_model_prefix"] = f"bpe_{config['vocab_size']}"
prompts = [
    "Once upon a time there was",
    "In a land far far away",
    "My name is Mariama, my favorite",
    '"Can cows fly?", Alice asked her mother.',
    "Alice was so tired when she got back home so she went",
]
device = torch.device("cuda")

if __name__ == "__main__":
    wandb.init(
        project=config["project"],
        name=config["name"],
        config=config,
    )

    model = TransformerModel(
        config["vocab_size"],
        config["emsize"],
        config["nhead"],
        config["d_hid"],
        config["nlayers"],
        config["dropout"],
    ).to(device)

    print(f"{number_of_weights(model)=}")

    encods_path = Path(f'encoded_stories_{config["sp_model_prefix"]}.npy')
    index_path = Path(f'index_{config["sp_model_prefix"]}.npy')
    sp_model = SentencePieceProcessor(model_file=config["sp_model_prefix"] + ".model")

    train_ds = TokenizedDataset(
        sp_model,
        config["max_length"],
        encods_path,
        index_path,
        train=True,
    )
    collate_fn = Collator(sp_model.pad_id())
    train_loader = inf_loop(
        DataLoader(
            train_ds,
            config["batch_size"],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=8,
            drop_last=True,
            pin_memory=True,
        )
    )
    train_loader = iter(train_loader)

    val_ds = TokenizedDataset(
        sp_model,
        config["max_length"],
        encods_path,
        index_path,
        train=False,
    )
    val_loader = DataLoader(
        val_ds,
        config["batch_size"],
        collate_fn=collate_fn,
        num_workers=8,
        drop_last=True,
        pin_memory=True,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        betas=(0.9, 0.95),
        weight_decay=config["weight_decay"],
    )
    scaler = torch.cuda.amp.GradScaler()

    step = 0
    for epoch in range(config["epochs"]):
        model.train()
        wandb.log({"epoch": epoch}, step=step)
        total_loss = 0
        total_grad = 0
        for batch_idx in tqdm(range(config["len_epoch"]), desc="train"):
            step += 1
            optimizer.zero_grad(set_to_none=True)
            for accum_step in range(config["accumulation_steps"]):
                batch = next(train_loader)
                src, tgt = batch["src"].to(device), batch["tgt"].to(device)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    with torch.backends.cuda.sdp_kernel(
                        enable_flash=True,
                        enable_math=False,
                        enable_mem_efficient=False,
                    ):
                        output = model(src)
                        output_flat = output.view(-1, config["vocab_size"])
                        loss = (
                            criterion(output_flat, tgt.view(-1))
                            / config["accumulation_steps"]
                        )

                total_loss += loss.item()
                scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            total_grad += get_grad_norm(model)
            scaler.step(optimizer)
            scaler.update()

            if (batch_idx + 1) % config["log_step"] == 0:
                wandb.log(
                    {"train_loss": total_loss / config["log_step"]},
                    step=step,
                )
                total_loss = 0
                wandb.log(
                    {"grad_norm": total_grad / config["log_step"]},
                    step=step,
                )
                total_grad = 0

        with torch.inference_mode():
            model.eval()
            total_loss = 0
            for batch in tqdm(val_loader, desc="validation", total=len(val_loader)):
                src, tgt = batch["src"].to(device), batch["tgt"].to(device)
                output = model(src)
                output_flat = output.view(-1, config["vocab_size"])
                loss = criterion(output_flat, tgt.view(-1))
                total_loss += loss
        wandb.log(
            {"val_loss": total_loss / len(val_loader)},
            step=step,
        )

        rows = {}
        for idx, prompt in enumerate(prompts):
            rows[idx] = {
                "prompt": prompt,
                "output": inference(model, sp_model, prompt=prompt),
            }
        wandb.log(
            {
                "generation": wandb.Table(
                    dataframe=pd.DataFrame.from_dict(rows, orient="index")
                )
            },
            step=step,
        )

        if (epoch + 1) % config["save_period"] == 0:
            torch.save(model.state_dict(), f"checkpoint_epoch{epoch}.pt")
