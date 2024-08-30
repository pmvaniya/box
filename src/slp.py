import os
import torch
import torch.nn.functional as F
from shutil import rmtree

from .tokenizer import load_tokens
from .utils import INFO_M, DANGER_M, SUCCESS_M
from .utils import title, panic


def train_slp(DATA_DIR):
    title("Training a Single Layer Perceptron Model")

    CSV_DIR = DATA_DIR / "csv_data"
    SLP_DIR = DATA_DIR / "slp"
    WEIGHTS_FILE = SLP_DIR / "weights.txt"

    if os.path.exists(SLP_DIR):
        rmtree(SLP_DIR)

    os.mkdir(SLP_DIR)

    csv_files = os.listdir(CSV_DIR)
    total_csv_files = len(csv_files)

    if total_csv_files < 1:
        print(f"{DANGER_M} CSV Directory is empty.")
        panic()

    print(f"{INFO_M} Found {total_csv_files} csv files in CSV Directory")

    tokens = load_tokens(DATA_DIR)
    vocab_size = len(tokens)

    print(f"{INFO_M} Starting Model Training")

    g = torch.Generator().manual_seed(64)
    W = torch.randn((vocab_size, vocab_size), generator=g, requires_grad=True)
    W.grad = None

    xs = []
    ys = []

    corpus_size = 50000
    counter = 0

    for csv_file in csv_files[:1]:
        filepath = CSV_DIR / csv_file

        with open(filepath) as inputfile:
            while True:
                row = inputfile.readline()

                if not row or row == "\n":
                    break

                x = [int(num) for num in row.split(", ")]

                xs += x[:-1]
                ys += x[1:]

                counter += len(x) - 1

                if counter > corpus_size:
                    break

    xs = xs[:counter]
    ys = ys[:counter]

    xs = torch.tensor(xs)
    ys = torch.tensor(ys)

    # don't judge the values, this is a playground
    epochs = 100
    learning_rate = 100

    for i in range(epochs):
        xenc = F.one_hot(xs, num_classes=vocab_size).float()
        logits = xenc @ W
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdim=True)
        loss = -probs[torch.arange(len(ys)), ys].log().mean()

        loss.backward()
        with torch.no_grad():
            W += -learning_rate * W.grad
            W.grad.zero_()

        # W.data += -1 * W.grad

        if i % 10 == 0:
            print(f"{INFO_M} epoch {i + 1} of {epochs}, loss: {loss.item():.4}")

    for variable in [xs, ys, xenc, logits, counts, probs]:
        del variable

    print(f"{SUCCESS_M} Model training completed.")
    print(f"{INFO_M} final loss: {loss.item():.4}")

    write_weights(WEIGHTS_FILE, W)

    del W
    del loss


def write_weights(filepath, W):
    weights = []
    outtext = ""

    for i in range(len(W)):
        weights.append([])

        for j in range(len(W[i])):
            weights[i].append(str(W[i][j].item()))

        outtext += ", ".join(weights[i])
        outtext += "\n"

    open("weights.txt", "w").write(outtext)

    try:
        with open(filepath, "w") as outfile:
            outfile.write(outtext)
            print(
                f"{SUCCESS_M} Successfully written model weights to {os.path.basename(filepath)}"
            )

    except Exception as e:
        print(f"{DANGER_M} Writing model weights to a text file failed. {e}")
        panic()
