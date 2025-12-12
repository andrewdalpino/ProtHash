# ProtHash

A protein language model that outputs amino acid sequence embeddings for use in clustering, classification, locality-sensitive hashing, and more. Distilled from the ESMC family of models with deep comprehension of protein structure, ProtHash produces contextual embeddings that align in vector space according to the sequences' atomic structure. Trained to mimic its ESMC teacher model, ProtHash achieves near perfect similarity to ESMC embeddings but at a greatly reduced computational cost.

## Key Features

- **Structurally-relevant embeddings**: Structurally similar proteins will show up nearby in the embedding space enabling downstream tasks such as clustering, classification, and locality-sensitive hashing based on atomic structure.

- **Blazing fast and efficient**: ProtHash uses only 3% of ESMC's parameters to achieve near perfect cosine similarity between the two embedding spaces.

- **Long context**: With a context window of 2048 amino acid tokens you can embed proteins with long sequences.

## Pretrained Models

| Name | Embedding Dimensionality | Attention Heads (Q/KV) | Encoder Layers | Total Params |
|---|---|---|---|---|
| [andrewdalpino/ProtHash-512-Tiny](https://huggingface.co/andrewdalpino/ProtHash-512-Tiny) | 512 | 16/4 | 4 | 13M |
| andrewdalpino/ProtHash-512 | 512 | 16/4 | 10 | 13M |

## Pretrained Example

First, you'll need the `prothash` and `esm` packages installed into your environment. We recommend using a virtual environment such as Python's `venv` module to prevent version conflicts with any system packages.

```sh
pip install prothash esm
```

Then, load the weights from HuggingFace Hub, tokenize a protein sequence, and pass it to the model. ProtHash adopts the ESM tokenizer as it's amino acids tokenization scheme. The output will be an embedding vector that can be used in downstream tasks such as comparing to other protein sequence embeddings, clustering, and near-duplicate detection.

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

x = torch.tensor(tokens, dtype=torch.int64).unsqueeze(0)

y_embed = model.embed(x)

print(y_embed)
```

## References

>- The UniProt Consortium, UniProt: the Universal Protein Knowledgebase in 2025, Nucleic Acids Research, 2025, 53, D609–D617.
>- T. Hayes, et al. Simulating 500 million years of evolution with a language model, 2024.
>- B. Zhang, et al. Root Mean Square Layer Normalization. 33rd Conference on Neural Information Processing Systems, NeurIPS 2019.
>- J. Ainslie, et al. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints, Google Research, 2023.
