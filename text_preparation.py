import argparse
import json
import os
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sentencepiece import SentencePieceProcessor, SentencePieceTrainer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm


class TokenizedDataset(Dataset):
    def __init__(
        self,
        sp_model,
        max_length,
        encoded_stories_path,
        index_path,
        train=False,
        test_size=0.05,
    ):
        super().__init__()
        self.sp_model = sp_model
        self.max_length = max_length
        self.encoded_stories = np.load(encoded_stories_path)
        _index = np.load(index_path)
        # index was shuffled so ordely split is okey
        if train:
            self.index = _index[: -int(test_size * len(_index))]
        else:
            self.index = _index[-int(test_size * len(_index)) :]

    def __getitem__(self, idx):
        start, end = self.index[idx]
        story = self.encoded_stories[start:end].tolist()
        encoded_story = [self.sp_model.bos_id()] + story + [self.sp_model.eos_id()]

        src = torch.tensor(encoded_story[:-1])
        tgt = torch.tensor(encoded_story[1:])

        story_length = min(len(src), self.max_length)

        return {
            "src": src[:story_length],
            "tgt": tgt[:story_length],
            "story_length": story_length,
        }

    def __len__(self):
        return len(self.index)


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch_list) -> Any:
        result_batch = {"src": [], "tgt": [], "story_length": []}

        for elem in batch_list:
            result_batch["src"].append(elem["src"])
            result_batch["tgt"].append(elem["tgt"])
            result_batch["story_length"].append(elem["story_length"])

        result_batch["story_length"] = torch.tensor(result_batch["story_length"])
        result_batch["src"] = pad_sequence(
            result_batch["src"], batch_first=True, padding_value=self.pad_id
        )
        result_batch["tgt"] = pad_sequence(
            result_batch["tgt"], batch_first=True, padding_value=self.pad_id
        )

        return result_batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for training.")
    parser.add_argument(
        "--vocab_size",
        "-vs",
        type=int,
        default=5000,
        help="Vocabularity size of sentencepiece tokenizer",
    )
    parser.add_argument(
        "--archive_path",
        "-ap",
        type=str,
        default="TinyStories_all_data.tar.gz",
        help="Path to TinyStories_all_data.tar.gz",
    )

    args = parser.parse_args()

    path_to_archive = Path(args.archive_path)

    if path_to_archive.exists():
        print(f"{path_to_archive} already exists.")
    else:
        print("Download TinyStories...")
        url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
        os.system(f'wget --quiet --show-progress "{url}"')

    path_to_data_folder = Path("TinyStories_all_data")

    if path_to_data_folder.exists():
        print(f"{path_to_archive} already exists.")
    else:
        print("Unpacking TinyStories")
        file = tarfile.open(path_to_archive)
        file.extractall(path_to_data_folder)

    merged_stories_path = Path("all_stories.txt")

    files = sorted(list(path_to_data_folder.glob("*.json")))

    if merged_stories_path.exists():
        print(f"{merged_stories_path} already exists.")
    else:
        print("Merge stories into one file...")

        for file in tqdm(files, total=len(files), desc="Processing files"):
            with open(file, "r") as f:
                data = json.load(f)
                with open(merged_stories_path, "a") as f:
                    for story in data:
                        text = story["story"].strip()
                        f.write(f"{text}\n")

    vocab_size: int = args.vocab_size
    normalization_rule_name: str = "nmt_nfkc_cf"
    model_type: str = "bpe"
    sp_model_prefix: str = f"{model_type}_{vocab_size}"

    desired_model_path = Path(f"{sp_model_prefix}.model")
    desired_vocab_path = Path(f"{sp_model_prefix}.vocab")

    if desired_model_path.exists() and desired_vocab_path.exists():
        print("SentencePieceTokenizer already exists.")
    else:
        print("Training SentencePieceTokenizer...")
        SentencePieceTrainer.train(
            input=merged_stories_path,
            vocab_size=vocab_size,
            model_type=model_type,
            model_prefix=sp_model_prefix,
            normalization_rule_name=normalization_rule_name,
            pad_id=3,
            num_threads=os.cpu_count(),
        )

    sp = SentencePieceProcessor(model_file=str(desired_model_path))

    encoded_stories = []
    index = []

    encods_path = Path(f"encoded_stories_{sp_model_prefix}.npy")
    index_path = Path(f"index_{sp_model_prefix}.npy")

    if encods_path.exists() and index_path.exists():
        print(f"{encods_path} already exists.")
    else:
        for file in tqdm(files, total=len(files), desc="Processing files"):
            with open(file, "r") as f:
                data = json.load(f)
                with open(merged_stories_path, "a") as f:
                    stories_batch = []
                    for story in data:
                        stories_batch.append(story["story"].strip())
                    encoded_stories_batch = sp.encode(stories_batch)
                    for encoded_story in encoded_stories_batch:
                        story_start = len(encoded_stories)
                        encoded_stories.extend(encoded_story)
                        story_end = len(encoded_stories)
                        index.append([story_start, story_end])
        encoded_stories = np.array(encoded_stories, dtype=np.int16)
        index = np.array(index, dtype=np.int32)
        print(f"{len(encoded_stories)=}")
        np.save(encods_path, encoded_stories)
        np.random.shuffle(index)
        np.save(index_path, index)
