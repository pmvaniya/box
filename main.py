from sys import argv
from pathlib import Path
from time import perf_counter

from src.utils import extract_open_web_text, title, random_bs_go, download_owt
from src.utils import INFO_M
from src.tokenizer import generate_bpe_mapping, convert_data_to_csv

CURR_DIR = Path(__file__).parent
DATA_DIR = CURR_DIR / "data"

if __name__ == "__main__":
    start_perf_counter = perf_counter()

    title("Start Of Program")

    print(f"{INFO_M} {' '.join(argv)}")

    if len(argv) > 1:

        if argv[1] == "extract":
            extract_open_web_text(DATA_DIR)

        elif argv[1] == "bpe" or argv[1] == "generate-tokens":
            generate_bpe_mapping(
                DATA_DIR, corpus_size=5000000, max_tokens=1000, min_freq=100
            )

        elif argv[1] == "convert":
            convert_data_to_csv(DATA_DIR)

        elif argv[1] == "download":
            download_owt(DATA_DIR)

        else:
            title("Something is Wrong")
            print(f"{INFO_M} I can feel it.")

    else:
        title("Did you RTFM?")
        print(f"{INFO_M} Go and RTFM")

    title("End Of Program")

    print(f"{INFO_M} Time taken: {(perf_counter() - start_perf_counter):.2f}s")
    random_bs_go()
