import os
import requests
import tarfile
import json

import numpy as np

from pathlib import Path
from shutil import rmtree

# the necessary functions will be called if set to True
DOWNLOAD_OWT = False
EXTRACT_OWT = False
CONVERT_OWT = False
TRAIN_LLM = True

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
shard_size = 100_000_000  # 100M tokens per shard


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

    for tar in tars[:2]:
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
    print("TRAIN LLM")


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
