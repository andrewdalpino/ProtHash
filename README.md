# ProtHash

A protein language model that outputs amino acid sequence embeddings for use in clustering, classification, locality-sensitive hashing, and more. Distilled from the [ESMC](https://www.evolutionaryscale.ai/blog/esm-cambrian) family of models with deep comprehension of protein structure, ProtHash produces contextual embeddings that align in vector space according to the sequences' atomic structure. Trained on the [SwissProt](https://huggingface.co/datasets/andrewdalpino/SwissProt-Gene-Ontology) dataset to mimic the activations of its ESMC teacher model, ProtHash embeddings have near perfect similarity to ESMC embeddings but at a greatly reduced computational cost.

## Key Features

- **Blazing fast and efficient**: ProtHash uses as few as 1.5% of its ESMC teacher's total parameters to achieve near-perfect cosine similarity between the two embedding spaces.

- **Structurally-relevant**: Structurally similar proteins will show up nearby in the embedding space enabling downstream tasks such as clustering, classification, and locality-sensitive hashing based on atomic structure.

- **Compatible with ESMC**: ProtHash can output embeddings in its native or ESMC teacher's dimensionality - allowing it to serve as both a faster drop-in replacement for ESMC embeddings and a more efficient compressed representation.

- **Quantization-ready**: With quantization-aware post-training, ProtHash allows you to quantize the weights of the model without losing similarity to the teacher's embedding space.

## Pretrained Models

| Name | Context Length | Embedding Dimensionality | Attention Heads (Q/KV) | Encoder Layers  | Total Params | Teacher Model | Teacher Dimensionality |
|---|---|---|---|---|---|---|---|
| [andrewdalpino/ProtHash-384-Tiny](https://huggingface.co/andrewdalpino/ProtHash-384-Tiny) | 2048 | 384 | 16/4 | 4 | 5M | esmc_300m | 960 |
| [andrewdalpino/ProtHash-384](https://huggingface.co/andrewdalpino/ProtHash-384) | 2048 | 384 | 16/4 | 10 | 11M | esmc_300m | 960 |
| [andrewdalpino/ProtHash-512-Tiny](https://huggingface.co/andrewdalpino/ProtHash-512-Tiny) | 2048 | 512 | 16/4 | 4 | 8.5M | esmc_600m | 1152 |
| [andrewdalpino/ProtHash-512](https://huggingface.co/andrewdalpino/ProtHash-512) | 2048 | 512 | 16/4 | 10 | 19M | esmc_600m | 1152 |

## Pretrained Example

First, you'll need the `prothash` and `esm` packages installed into your environment. We recommend using a virtual environment such as Python's `venv` module to prevent version conflicts with any system packages.

```sh
pip install prothash esm
```

Then, load the weights from HuggingFace Hub, tokenize a protein sequence, and pass it to the model. ProtHash adopts the ESM tokenizer as it's amino acids tokenization scheme which consists of a vocabulary of 33 amino acid and special tokens. The output will be an embedding vector that can be used in downstream tasks such as comparing to other protein sequence embeddings, clustering, and near-duplicate detection.

```python
import torch

from esm.tokenization import EsmSequenceTokenizer

from prothash.model import ProtHash

tokenizer = EsmSequenceTokenizer()

model_name = "andrewdalpino/ProtHash-512-Tiny"

model = ProtHash.from_pretrained(model_name)

sequence = input("Enter a sequence: ")

out = tokenizer(sequence, max_length=2048)

tokens = out["input_ids"]

# Input is a [1, T] tensor of token indices. 
x = torch.tensor(tokens, dtype=torch.int64).unsqueeze(0)

# Output the sequence embedding in native dimensionality.
y_embed_native = model.embed_native(x).squeeze(0)

print(y_embed_native.shape)

# Output a drop-in replacement for the teacher's embeddings.
y_embed_teacher = model.embed_teacher(x).squeeze(0)

print(y_embed_teacher.shape)
```

## References

>- The UniProt Consortium, UniProt: the Universal Protein Knowledgebase in 2025, Nucleic Acids Research, 2025, 53, D609–D617.
>- T. Hayes, et al. Simulating 500 million years of evolution with a language model, 2024.
>- B. Zhang, et al. Root Mean Square Layer Normalization. 33rd Conference on Neural Information Processing Systems, NeurIPS 2019.
>- J. Ainslie, et al. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints, Google Research, 2023.
>- T. Kim, et al. Comparing Kullback-Leibler Divergence and Mean Squared Error Loss in Knowledge Distillation, 2021.
