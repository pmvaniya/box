import os
from json import dump, load
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
        text += read_file(textfile_path) + "\n"

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
    token_counter = 0
    frequencies = {}
    itos = {}

    token_ids["<<OutOfVocabulary>>"] = 0
    token_ids["<<Start>>"] = 1
    token_ids["<<End>>"] = 2

    token_counter += 3

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


def load_tokens(DATA_DIR):
    try:
        token_path = DATA_DIR / "bpe" / "token_ids.json"
        return load(open(token_path, "r"))

    except Exception as e:
        print(f"{DANGER_M} Error while reading token_ids.json. {e}")
        panic()


def convert_data_to_csv(DATA_DIR):
    title("Converting all text data to csv files")

    EXTRACTED_DIR = DATA_DIR / "extracted"
    CSV_DIR = DATA_DIR / "csv_data"

    check_dir_exists(DATA_DIR, "Data", panic_mode=True)
    check_dir_exists(EXTRACTED_DIR, "Extracted", panic_mode=True)

    if os.path.exists(CSV_DIR):
        rmtree(CSV_DIR)

    os.mkdir(CSV_DIR)

    extracted_folders = sorted(os.listdir(EXTRACTED_DIR))
    total_extracted_folders = len(extracted_folders)

    if total_extracted_folders < 1:
        print(f"{DANGER_M} Extracted directory is empty.")
        panic()

    print(
        f"{INFO_M} {total_extracted_folders} directories found in extracted directory."
    )

    token_ids = load_tokens(DATA_DIR)
    print(f"{INFO_M} {len(token_ids)} tokens loaded.")

    textfile_counter = 0
    csv_file_counter = 1
    csv_file_path = CSV_DIR / ("csv_data_%06d.csv" % (csv_file_counter))
    csv_file = open(csv_file_path, "a")

    for extracted_folder in extracted_folders:
        extracted_folder_path = EXTRACTED_DIR / extracted_folder
        textfiles = os.listdir(extracted_folder_path)
        print(f"{INFO_M} {len(textfiles)} textfiles found in {extracted_folder}")

        for textfile in textfiles:
            textfile_path = extracted_folder_path / textfile
            text = read_file(textfile_path)
            encoded_text = encode_text(text, token_ids)
            csv_file.write(", ".join([str(x) for x in encoded_text]) + "\n")

            textfile_counter += 1

            if textfile_counter % 10000 == 0:
                csv_file_counter += 1
                csv_file_path = CSV_DIR / ("csv_data_%06d.csv" % (csv_file_counter))
                csv_file = open(csv_file_path, "a")

        print(f"{SUCCESS_M} Converted all textfiles in {extracted_folder} to csv.")

    print(
        f"{INFO_M} Converted {textfile_counter} text files to {csv_file_counter} csv files."
    )


def encode_text(text, stoi):
    text_length = len(text)
    max_token_length = 10  # The maximum token length in the stoi

    encoded_text = []
    encoded_text.append(stoi["<<Start>>"])

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
            encoded_text.append(stoi["<<OutOfVocabulary>>"])
            i += 1

    encoded_text.append(stoi["<<End>>"])

    return encoded_text
