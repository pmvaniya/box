import os
import tarfile
from shutil import rmtree
from pathlib import Path

# this program expects all the tar files in a directory called OpenWebText
ROOT_DIR = Path(__file__).parent
TAR_DIR = ROOT_DIR / "OpenWebText"
DATA_DIR = ROOT_DIR / "data"

INFO_M = "\033[34m  INFO    \033[0m"
WARN_M = "\033[33m  WARN    \033[0m"
DANGER_M = "\033[31m DANGER   \033[0m"
SUCCESS_M = "\033[32m  SUCC    \033[0m"
QUESTION_M = "\033[35m  QUES    \033[0m"

# a directory created on extracting an OpenWebText tar file
openwebtext_path = DATA_DIR / "openwebtext"


def create_data_dir():
    try:
        os.mkdir(DATA_DIR)
        print(INFO_M + "Created new data directory.")
    except Exception as e:
        print(DANGER_M, e)
        exit()


def delete_data_dir():
    print(DANGER_M + "Deleting data directory.")
    try:
        rmtree(DATA_DIR)
    except Exception as e:
        print(DANGER_M, e)
        exit()


def exit_program():
    print(INFO_M + "Exiting Program")
    exit()


# check if OpenWebText directory exists or not
if os.path.isdir(TAR_DIR):
    print(INFO_M + "OpenWebText directory found.")

else:
    print(INFO_M + "OpenWebText directory not found.")
    exit_program()

# create data directory if it does not exist
if os.path.isdir(DATA_DIR):
    print(WARN_M + "Data directory already exists.")
    choice = input(QUESTION_M + "Would you like to delete it? (y/N): ")

    if choice in ["y", "Y"]:
        delete_data_dir()
        create_data_dir()
    else:
        exit_program()

else:
    create_data_dir()


tar_files = sorted(os.listdir(TAR_DIR))

if len(tar_files) == 0:
    print(DANGER_M + "OpenWebText directory is empty.")
    exit_program()


for tar_file in tar_files:
    tar_file_path = TAR_DIR / tar_file
    new_dataset_path = DATA_DIR / tar_file[:-4]

    try:
        print(INFO_M + "Extracting", tar_file)
        tarfile.open(tar_file_path, "r").extractall(DATA_DIR, filter="data")
        sub_tars = os.listdir(openwebtext_path)

        for sub_tar in sub_tars:
            sub_tar_path = openwebtext_path / sub_tar
            tarfile.open(sub_tar_path, "r").extractall(new_dataset_path, filter="data")

        rmtree(openwebtext_path)

    except Exception as e:
        print(DANGER_M, e)
        exit_program()
