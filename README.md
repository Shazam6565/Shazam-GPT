# Transformer Text Completion Model: README

## Overview

This project involves the implementation and optimization of a text completion model based on the Transformer architecture, as detailed in the paper "Attention Is All You Need" presented at the 31st Conference on Neural Information Processing Systems (NIPS 2017). The aim was to systematically explore the impact of varying learning rates and training epochs on the model's performance, targeting the reduction of validation loss and the improvement of text coherence and fluency.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Experiments](#experiments)
5. [Results](#results)
6. [Contributions](#contributions)
7. [License](#license)
8. [References](#references)

## Introduction

Transformers have revolutionized AI research and applications, especially in natural language processing (NLP), by utilizing an attention mechanism to capture long-range dependencies and contextual information efficiently. This project implements a text completion model using the Transformer architecture and examines the effects of different learning rates on the model's performance.

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/Shazam6565/Shazam-GPT.git
   cd Shazam-GPT
   ```

2. Create a virtual environment and activate it:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

1. Prepare the dataset by downloading "The Stock Exchange" by Charles Duguid from Project Gutenberg:
   ```sh
   wget https://www.gutenberg.org/ebooks/59042 -O dataset.txt
   ```

2. Preprocess the data:
   ```sh
   python preprocess.py --input dataset.txt --output preprocessed_data.pkl
   ```

3. Train the model:
   ```sh
   python train.py --config config.yaml
   ```

4. Evaluate the model:
   ```sh
   python evaluate.py --model checkpoint.pth --data preprocessed_data.pkl
   ```

## Experiments

### Tools and Resources

- **Programming Language:** Python 3.10
- **Deep Learning Framework:** PyTorch or TensorFlow
- **Tokenizer:** Hugging Face Tokenizer or NLTK
- **Libraries:** NumPy, Matplotlib, Seaborn, Scikit-learn
- **Hardware:** NVIDIA A100 GPUs with 64GB memory
- **IDE:** Jupyter Notebook, Visual Studio Code
- **Operating System:** MacOS
- **Distributed Training:** PyTorch’s DistributedDataParallel or TensorFlow’s tf.distribute.Strategy
- **Logging and Monitoring:** TensorBoard
- **Version Control:** Git

### Experimental Setup

Seven experiments were conducted with different combinations of learning rates and training-validation data splits:
- Learning Rates: 1e-6, 1e-5, 1e-4, 2e-4, 3e-4
- Data Splits: 70-30, 80-20

### Performance Metrics

1. **Perplexity Score:** Evaluated on the validation set.
2. **Validation Loss:** Monitored to gauge the model's generalization.
3. **Training Loss:** Monitored to ensure convergence.

## Results

The optimal configuration was found with a learning rate of 2e-4 and a data split of 70-30, which minimized the validation loss and perplexity score:
- **Validation Loss:** 0.092
- **Perplexity Score:** 1.097


## License

This project is licensed under the MIT License. See the LICENSE file for details.

## References

1. Vaswani, A., et al. (2017). "Attention Is All You Need." Advances in Neural Information Processing Systems, 30.
2. Bahdanau, D., et al. (2014). "Neural Machine Translation by Jointly Learning to Align and Translate." arXiv preprint arXiv:1409.0473.
3. Sutskever, I., et al. (2014). "Sequence to Sequence Learning with Neural Networks." Advances in Neural Information Processing Systems, 27.

For further details, refer to the [final report](Transformer_Implementation_Report.pdf) included in the repository.
