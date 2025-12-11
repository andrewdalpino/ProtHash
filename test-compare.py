from argparse import ArgumentParser

import torch

from esm.tokenization import EsmSequenceTokenizer
from esm.models.esmc import ESMC

from src.prothash.model import ProtHash

from data import SwissProt


def main():
    parser = ArgumentParser(
        description="Compare the embeddings of ProtHash versus ESMC."
    )

    parser.add_argument(
        "--checkpoint_path", default="./checkpoints/checkpoint.pt", type=str
    )
    parser.add_argument("--teacher_model_name", type=str, default="esmc_300m")
    parser.add_argument("--min_sequence_length", default=1, type=int)
    parser.add_argument("--max_sequence_length", default=2048, type=int)
    parser.add_argument("--device", default="cpu", type=str)

    args = parser.parse_args()

    if "cuda" in args.device and not torch.cuda.is_available():
        raise RuntimeError("Cuda is not available.")

    tokenizer = EsmSequenceTokenizer()

    dataset = SwissProt(
        tokenizer=tokenizer,
        min_sequence_length=args.min_sequence_length,
        max_sequence_length=args.max_sequence_length,
    )

    teacher = ESMC.from_pretrained(args.teacher_model_name)

    teacher = teacher.to(args.device)

    teacher.eval()

    print("Teacher model loaded successfully")

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=True)

    student = ProtHash(**checkpoint["model_args"])

    student.remove_adapter_head()

    student.load_state_dict(checkpoint["model"])

    student = student.to(args.device)

    student.eval()

    print("Model checkpoint loaded successfully")


if __name__ == "__main__":
    main()
