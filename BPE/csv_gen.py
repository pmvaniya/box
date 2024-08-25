import os
from pathlib import Path
from json import dump

# some useful constants
INFO_M = "\033[34m  INFO    \033[0m"
WARN_M = "\033[33m  WARN    \033[0m"
DANGER_M = "\033[31m DANGER   \033[0m"
SUCCESS_M = "\033[32m  SUCC    \033[0m"
QUESTION_M = "\033[35m  QUES    \033[0m"

print(SUCCESS_M, "Start of Program")

curr_dir = Path(__file__).parent

owt_dir = curr_dir.parent / "data" / "urlsf_subset00"
csv_path = curr_dir / "encoded_textfiles.csv"
tokens_path = curr_dir / "token_ids.json"
file_metadata_path = curr_dir / "file_metadata.json"

token_ids = {}
file_data = {}
token_counter = 0
file_counter = 1

open(csv_path, "w").close()  # clear contents of the file
csv_file = open(csv_path, "a")

textfiles = os.listdir(owt_dir)
total_files = len(textfiles)

for textfile in textfiles:
    print(INFO_M, "file %d of %d" % (file_counter, total_files))
    file_counter += 1

    tokens = []
    filepath = owt_dir / textfile
    text = open(filepath, "r").read()

    for char in text:
        if ord(char) > 127:
            continue  # skip for any non ascii character

        try:
            tokens.append(token_ids[char])
        except:
            tokens.append(token_counter)
            token_ids[char] = token_counter
            token_counter += 1

    file_data[textfile] = [len(text), len(tokens)]
    csv_file.write(", ".join([str(token) for token in tokens]) + "\n")

csv_file.close()

# write token_ids to a json file
with open(tokens_path, "w") as outfile:
    dump(token_ids, outfile, indent=4)
    outfile.close()
    print(SUCCESS_M, "Successfully wrote token ids to", tokens_path)

# write file metadata to a json file
with open(file_metadata_path, "w") as outfile:
    dump(file_data, outfile, indent=4)
    outfile.close()
    print(SUCCESS_M, "Successfully wrote file metadata to", tokens_path)

print(SUCCESS_M, "End of Program")
