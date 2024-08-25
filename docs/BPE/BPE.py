from pathlib import Path
from json import dump

# set up path
curr_dir = Path(__file__).parent
textfile = curr_dir / "GreatExpectations.txt"

max_tokens = 264

try:
    text = open(textfile, "r", encoding="utf-8").read()

    # since we are reading from a book that has new lines, we can replace them with a space
    # this is only done to prevent \n in byte pairs, it would not be recommended for training data
    text = text.replace("\n", " ")

    # very unlikely, but it is possible that the text file may not contain one of the standard characters, so we just append that to the text
    text += "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789\n"

except Exception as e:
    print("Reading from source file failed.", e)
    exit()

vocab = sorted(list(set(text)))
vocab_size = len(vocab)

token_dict = {}
token_counter = 0


def encode(text):
    encoded = []

    for char in text:
        encoded.append(
            next(
                (token_id for token_id, token in token_dict.items() if token == char),
                None,
            )
        )

    return encoded


def decode(encoded):
    decoded = ""

    for code in encoded:
        decoded += token_dict[code]

    return decoded


def get_frequencies(tokens):
    frequencies = {}

    for i in range(len(tokens) - 1):
        try:
            frequencies[(tokens[i], tokens[i + 1])] += 1
        except:
            frequencies[(tokens[i], tokens[i + 1])] = 1

    return frequencies


def merge_pairs(tokens, pair, token_id):
    i = 0
    new_tokens = []

    while i < len(tokens) - 1:  # prevent out of range index
        if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            new_tokens.append(token_id)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    new_tokens.append(tokens[-1])  # add the last indexwhile token_counter < 30

    return new_tokens


for char in vocab:
    token_dict[token_counter] = char
    token_counter += 1

print("initial tokens:", token_dict)

tokens = encode(text)

# continue merging until a desired vocab size is achieved
while token_counter < max_tokens:
    frequencies = get_frequencies(tokens)
    highest_pair = max(frequencies, key=frequencies.get)

    if frequencies[highest_pair] == 1:  # break if no more paires can be merged
        break

    tokens = merge_pairs(tokens, highest_pair, token_counter)
    token_dict[token_counter] = decode(highest_pair)  # add new token to the dictionary
    token_counter += 1

print("\nfinal tokens:", token_dict)


# write to a json file
output_file = curr_dir / "tokens.json"

try:
    with open(output_file, "w", encoding="utf-8") as outfile:
        dump(token_dict, outfile, indent=4)
        print("\nTokens successfully saved to tokens.json file.")

except Exception as e:
    print("\nWriting tokens to output file failed.", e)
    exit()
