import os
from json import dump
from shutil import rmtree
from collections import Counter

from .utils import check_dir_exists, title, panic, read_file
from .utils import INFO_M, DANGER_M, SUCCESS_M


def generate_bpe_mapping(DATA_DIR, corpus_size=500000, max_tokens=2056, min_freq=100):
    title("Generating Mapping using BPE")

    EXTRACTED_DIR = DATA_DIR / "extracted"
    BPE_DIR = DATA_DIR / "bpe"
    TOKEN_ID_FILE = BPE_DIR / "token_ids.json"
    FREQUENCIES_FILE = BPE_DIR / "frequencies.json"

    check_dir_exists(DATA_DIR, "Data", panic_mode=True)
    check_dir_exists(EXTRACTED_DIR, "Extracted", panic_mode=True)

    if os.path.exists(BPE_DIR):
        rmtree(BPE_DIR)

    os.mkdir(BPE_DIR)

    extracted_folders = sorted(os.listdir(EXTRACTED_DIR))
    total_extracted_folders = len(extracted_folders)

    if total_extracted_folders < 1:
        print(f"{DANGER_M} Extracted directory is empty.")
        panic()

    textfolder = EXTRACTED_DIR / extracted_folders[0]
    textfiles = os.listdir(textfolder)
    total_textfiles = len(textfiles)

    if total_textfiles < 1:
        print(f"{DANGER_M} {extracted_folders[0]} directory is empty.")
        panic()

    text = ""

    for textfile in textfiles:
        textfile_path = textfolder / textfile
        text += read_file(textfile_path)

        if len(text) > corpus_size:
            break

    token_ids, frequencies = bpe(text, max_tokens, min_freq)

    write_token_and_freq(TOKEN_ID_FILE, FREQUENCIES_FILE, token_ids, frequencies)


def bpe(text, max_tokens, min_freq):
    print(
        f"{INFO_M} Corpus size: {len(text)} characters, Max no of Tokens: {max_tokens}, Min frequency: {min_freq}"
    )

    tokens = list(text)
    char_set = sorted(set(tokens))
    char_count = Counter(tokens)

    token_ids = {}
    token_counter = 1
    frequencies = {}
    itos = {}

    token_ids["<OutOfVocabulary>"] = 0

    for char in char_set:
        if char_count[char] >= min_freq:
            token_ids[char] = token_counter
            frequencies[char] = char_count[char]
            itos[token_counter] = char
            token_counter += 1

        else:
            tokens = list(filter(lambda x: x != char, tokens))

    tokens = [token_ids[i] for i in tokens]

    print(f"{INFO_M} {token_counter} initial unique character tokens.")
    print(f"{INFO_M} Now starting merging of tokens")

    while token_counter < max_tokens:
        highest_pair, highest_count = get_most_frequent_pair(tokens)

        if 1 <= highest_count < min_freq:
            break

        tokens = merge_pair(tokens, highest_pair, token_counter)
        itos[token_counter] = itos[highest_pair[0]] + itos[highest_pair[1]]
        token_ids[itos[token_counter]] = token_counter
        frequencies[itos[token_counter]] = highest_count

        print(
            f"{INFO_M} Merged ({repr(itos[highest_pair[0]])}, {repr(itos[highest_pair[1]])}) -> {repr(itos[token_counter])}"
        )
        token_counter += 1

    print(f"{INFO_M} Generated vocabulary of {token_counter} tokens.")

    return token_ids, frequencies


def merge_pair(tokens, pair, new_token_id):
    new_tokens = []
    i = 0

    while i < len(tokens) - 1:
        if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            new_tokens.append(new_token_id)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1

    new_tokens.append(tokens[-1])

    return new_tokens


def get_most_frequent_pair(tokens):
    frequencies = {}

    for id1, id2 in zip(tokens, tokens[1:]):
        try:
            frequencies[(id1, id2)] += 1
        except:
            frequencies[(id1, id2)] = 1

    highest_pair = max(zip(frequencies.values(), frequencies.keys()))[1]

    return highest_pair, frequencies[highest_pair]


def write_token_and_freq(TOKEN_ID_FILE, FREQUENCIES_FILE, token_ids, frequencies):

    frequencies = dict(
        sorted(frequencies.items(), key=lambda item: item[1], reverse=True)
    )

    try:
        with open(TOKEN_ID_FILE, "w") as outfile:
            dump(token_ids, outfile, indent=4)
            print(f"{SUCCESS_M} Successfully wrote token ids to a json file.")

        with open(FREQUENCIES_FILE, "w") as outfile:
            dump(frequencies, outfile, indent=4)
            print(f"{SUCCESS_M} Successfully wrote the frequencies to a json file.")

    except Exception as e:
        print(f"{DANGER_M} Writing token ids / frequencies to a json file failed. {e}")
        panic()
