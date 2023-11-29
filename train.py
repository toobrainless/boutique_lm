from itertools import repeat
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from model import TransformerModel
from text_preparation import Collator, TokenizedDataset

device = torch.device("cuda")


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
    from torch.distributions import Categorical

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
    "emsize": 256,  # embedding dimension
    "d_hid": 64,  # dimension of the feedforward network model in ``nn.TransformerEncoder``
    "nlayers": 8,  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
    "nhead": 16,  # number of heads in ``nn.MultiheadAttention``
    "dropout": 0.1,  # dropout probability
    "max_length": 256,
    "batch_size": 352,
    "lr": 5e-4,
    "weight_decay": 0,
    "epochs": 100,
    "len_epoch": 10000,
    "log_step": 100,
    "accumulation_steps": 1,
    "project": "boutique_lm",
    "name": "test_table",
    "save_period": 1,
}
config["sp_model_prefix"] = f"bpe_{config['vocab_size']}"

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

    ds = TokenizedDataset(
        config["sp_model_prefix"], config["max_length"], encods_path, index_path
    )
    collate_fn = Collator(ds.sp_model.pad_id())

    loader = inf_loop(
        DataLoader(
            ds,
            config["batch_size"],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=8,
            drop_last=True,
            pin_memory=True,
        )
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        betas=(0.9, 0.95),
        weight_decay=config["weight_decay"],
    )
    scaler = torch.cuda.amp.GradScaler()
    loader = iter(loader)

    step = 1
    model.train()
    for epoch in range(config["epochs"]):
        wandb.log({"epoch": epoch}, step=step)
        total_loss = 0
        total_grad = 0
        for batch_idx in tqdm(range(config["len_epoch"]), desc="train"):
            optimizer.zero_grad(set_to_none=True)
            for accum_step in range(config["accumulation_steps"]):
                batch = next(loader)
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

                scaler.scale(loss).backward()
                total_loss += loss.item()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            total_grad += get_grad_norm(model)
            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()

            if (batch_idx + 1) % config["log_step"] == 0:
                step = epoch * config["len_epoch"] + batch_idx + 1
                wandb.log(
                    {"loss": total_loss / config["log_step"]},
                    step=step,
                )
                total_loss = 0
                wandb.log(
                    {"grad_norm": total_grad / config["log_step"]},
                    step=step,
                )
                total_grad = 0

        step = (epoch + 1) * config["len_epoch"]
        rows = {}
        for idx, prompt in enumerate(
            ["Hello", "Once upon a time there was", "In a land far far away"]
        ):
            rows[idx] = {
                "prompt": prompt,
                "output": inference(model, ds.sp_model, prompt=prompt),
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
