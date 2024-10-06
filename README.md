# Box: On-Device LLM for Smartphones

```DO NOT REFER TO THIS README FOR NOW. WILL BE UPDATED LATER SOON.```

`Box` is an on-device Large Language Model (LLM) designed specifically for smartphones. It operates entirely on-device, ensuring all computations are done locally without the need for server-side processing.

<br />

## Overview

`Box` started as a passion project and remains a work in progress. The goal is to create a powerful LLM that can run directly on smartphones, eliminating the need for server-side computations. This repository contains the LLM itself, while the corresponding Android app for `Box` can be found [here](https://github.com/pranav-vaniya/box-android-app). The model is trained using the [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext) dataset.

<br />

## How to Use

1. **Install Dependencies**
   - Make any necessary changes to `main.py`.
   - In the parent directory, run the following command to install the required Python dependencies:

     ```bash
     pip3 install -r requirements.txt
     ```

2. **Download Dataset**
   - Download the OpenWebText dataset with:

     ```bash
     python3 main.py download
     ```

3. **Extract Archives**
   - Extract the downloaded archives by running:

     ```bash
     python3 main.py extract
     ```

4. **Generate Vocabulary**
   - Create language vocabulary using Byte Pair Encoding (BPE):

     ```bash
     python3 main.py bpe
     ```

5. **Convert Text to CSV**
   - Convert the extracted text files into CSV files with encoded text:

     ```bash
     python3 main.py convert
     ```

6. **Train the Model**
   - Train the model. For example, to train a Single Layer Perceptron (SLP) model:

     ```bash
     python3 main.py train slp
     ```

   - You can replace `slp` with one of the following options (currently under development):
     - `mlp` for a Multilayer Perceptron Model (not available yet)
     - `rnn` for a Recurrent Neural Network (not available yet)

<br />

## Acknowledgements

I would like to extend my heartfelt gratitude to the following content creators and websites for their invaluable resources and inspiration:

- [Andrej Karpathy](https://www.youtube.com/@AndrejKarpathy)
- Any many more ...

The code has been highly inspired from the above mentioned mediums and i am greatly indebted to them.

<br />

## Thanks

Thank you for checking out `Box`. Check out [Box Android App](https://github.com/pranav-vaniya/box-android-app).

Happy coding!
