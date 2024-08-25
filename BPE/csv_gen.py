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

data_dir = curr_dir.parent / "data"
tokens_path = curr_dir / "token_ids.json"
out_dir = curr_dir / "data"

if not os.path.exists(out_dir):
    os.mkdir(out_dir)
else:
    os.rename(out_dir, curr_dir / "data-old")

token_ids = {}
file_data = {}
token_counter = 0
file_counter = 1
subset_counter = 1
partition_counter = 1

subset_dirs = os.listdir(data_dir)
total_subset_dirs = len(subset_dirs)

csv_path = out_dir / ("encoded_textfiles_%02d.csv" % (partition_counter))
file_metadata_path = out_dir / ("file_metadata_%02d.json" % (partition_counter))
open(csv_path, "w").close()  # clear contents of the file
csv_file = open(csv_path, "a")

for subset_dir in subset_dirs:
    textfiles = os.listdir(data_dir / subset_dir)
    total_files = len(textfiles)

    for textfile in textfiles:
        tokens = []
        filepath = data_dir / subset_dir / textfile
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

        print(
            INFO_M,
            "%s - %s (folder %d of %d, file %d of %d)"
            % (
                subset_dir,
                textfile,
                subset_counter,
                total_subset_dirs,
                file_counter,
                total_files,
            ),
        )

        if file_counter % 100000 == 0:
            # write file metadata to a json file
            with open(file_metadata_path, "w") as outfile:
                dump(file_data, outfile, indent=4)
                outfile.close()
                print(
                    SUCCESS_M, "Successfully wrote file metadata to", file_metadata_path
                )

            file_data = {}
            csv_file.close()

            partition_counter += 1
            csv_path = out_dir / ("encoded_textfiles_%02d.csv" % (partition_counter))
            file_metadata_path = out_dir / (
                "file_metadata_%02d.json" % (partition_counter)
            )
            open(csv_path, "w").close()  # clear contents of the file
            csv_file = open(csv_path, "a")

        file_counter += 1

    subset_counter += 1

csv_file.close()

# write file metadata to a json file
with open(file_metadata_path, "w") as outfile:
    dump(file_data, outfile, indent=4)
    outfile.close()
    print(SUCCESS_M, "Successfully wrote file metadata to", file_metadata_path)

# write token_ids to a json file
with open(tokens_path, "w") as outfile:
    dump(token_ids, outfile, indent=4)
    outfile.close()
    print(SUCCESS_M, "Successfully wrote token ids to", tokens_path)

print(SUCCESS_M, "End of Program")
