import os
import tarfile
from pathlib import Path
from shutil import rmtree
from random import choice

SUCCESS_M = "\033[32m  SUCC  \033[0m"  # green
INFO_M = "\033[34m  INFO  \033[0m"  # blue
WARN_M = "\033[33m  WARN  \033[0m"  # yellow
DANGER_M = "\033[31m DANGER \033[0m"  # red
QUESTION_M = "\033[35m  QUES  \033[0m"  # purple


def panic():
    print(f"{DANGER_M} Program terminated unexpectedly.")
    exit()


def title(text):
    text = f"> {text}"
    print(f"\n\033[1m{text}\033[0m")


def check_dir_exists(dir_path, name="Directory", panic_mode=False):
    if os.path.exists(dir_path):
        print(f"{INFO_M} {name} directory found.")

    else:
        print(f"{DANGER_M} {name} directory not found.")

        if panic_mode:
            panic()


def read_file(filename):
    try:
        file = open(filename, "r")
        contents = file.read()
        file.close()

        return contents

    except Exception as e:
        print(f"{DANGER_M} {e}")
        panic()


def extract_tar_file(source_path, destination_path):
    with tarfile.open(source_path, "r") as tar:
        tar.extractall(destination_path)


def extract_open_web_text(DATA_DIR):
    title("Extract OpenWebText")

    OWT_TAR_DIR = DATA_DIR / "OpenWebText"
    STAGE1_DIR = DATA_DIR / "stage1"
    STAGE1_TEMP_DIR = STAGE1_DIR / "openwebtext"
    EXTRACTED_DIR = DATA_DIR / "extracted"

    check_dir_exists(DATA_DIR, "Data", panic_mode=True)
    check_dir_exists(OWT_TAR_DIR, "OpenWebText", panic_mode=True)

    if os.path.exists(STAGE1_DIR):
        rmtree(STAGE1_DIR)

    if os.path.exists(EXTRACTED_DIR):
        rmtree(EXTRACTED_DIR)

    subset_tars = sorted(os.listdir(OWT_TAR_DIR))
    total_subset_tars = len(subset_tars)
    subset_counter = 1

    if total_subset_tars == 0:
        print(f"{DANGER_M} OpenWebText directory is empty.")
        panic()
    else:
        print(f"{INFO_M} OpenWebText directory contains {total_subset_tars} archives.")

    print(f"{INFO_M} Starting archive extractions")

    for subset_tar in subset_tars[:1]:
        print(
            f"{INFO_M} Extracting {subset_tar} (archive {subset_counter} of {total_subset_tars})"
        )
        subset_tar_path = OWT_TAR_DIR / subset_tar
        extraction_path = EXTRACTED_DIR / subset_tar[:-4]  # remove '.tar' from name
        extract_tar_file(subset_tar_path, STAGE1_DIR)

        for child_tar in os.listdir(STAGE1_TEMP_DIR)[:25]:
            child_tar_path = STAGE1_TEMP_DIR / child_tar
            extract_tar_file(child_tar_path, extraction_path)

        rmtree(STAGE1_TEMP_DIR)
        subset_counter += 1

    rmtree(STAGE1_DIR)
    print(f"{SUCCESS_M} All archives extracted successfully.")


def random_bs_go():
    # some random quotes I found on the internet
    bs = [
        "It's scary because it's new, not because you are incapable.",
        "You lack focus, not potential. You lack consistency, not ability. You lack strategy not resources. You lack discipline, not opportunities.",
        "History doesn't talk about losers.",
        "Until death, all defeat is psychological.",
        "Lessons in life will be repeated until they are learned.",
        "The fears we don't face become our limits.",
        "Don't die with regrets.",
        "If it is humanly possible, consider it to be within your reach.",
        "We judge others by their actions while we judge ourselves by our intentions.",
        "The magic lies within you.",
    ]

    print(f"{INFO_M} {choice(bs)}")


if __name__ == "__main__":
    pass
