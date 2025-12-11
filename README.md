# ProtHash

A protein language model that outputs amino acid sequence embeddings for use in clustering, classification, locality-sensitive hashing, and more. Distilled from the ESMC family of models with deep comprehension of protein structure, ProtHash produces contextual embeddings that align in vector space according to the sequences' atomic structure. Trained to mimic its ESMC teacher model, ProtHash achieves near perfect similarity to ESMC embeddings but at a greatly reduced computational cost.

## Key Features

- **Structurally-relevant embeddings**: Structurally similar proteins will show up nearby in the embedding space enabling downstream tasks such as clustering, classification, and locality-sensitive hashing based on atomic structure.

- **Blazing fast and efficient**: ProtHash requires only 5% of ESMC's parameters to achieve near perfect cosine similarity between the two embedding spaces.

- **Long context**: With a context window of 2048 you can embed proteins with very long amino acid sequences.

## References

>- The UniProt Consortium, UniProt: the Universal Protein Knowledgebase in 2025, Nucleic Acids Research, 2025, 53, D609–D617.
>- T. Hayes, et al. Simulating 500 million years of evolution with a language model, 2024.
