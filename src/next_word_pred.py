import sys
import torch
from fairseq.models.transformer_lm import TransformerLanguageModel
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import fastBPE


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir")
    parser.add_argument("--data_dir")
    parser.add_argument("--test_file")
    parser.add_argument("--top_k_out_file")
    parser.add_argument("--codes_path")
    parser.add_argument("--vocab_path")
    parser.add_argument("--top_k", type=int, default=25)
    args = parser.parse_args()

    # custom_lm = TransformerLanguageModel.from_pretrained(
    #     "/home/thclark/diachron-lm/models/1990",
    #     data_name_or_path="/home/thclark/diachron-lm/data/coha/lm_data/1990/en-bin",
    #     checkpoint_file="checkpoint_best.pt",
    # )

    custom_lm = TransformerLanguageModel.from_pretrained(
        args.checkpoint_dir,
        data_name_or_path=args.data_dir,
        checkpoint_file="checkpoint_best.pt",
    )
    custom_lm.eval()

    bpe = fastBPE.fastBPE(args.codes_path, args.vocab_path)

    # read data
    with open(args.test_file, "r") as f:
        lines = f.read().splitlines()
        lines = list(filter(lambda x: not x.startswith("#"), lines))
    lines = bpe.apply(lines)

    # get hidden representations
    all_embeddings, top_k_data = [], []
    for l in lines:
        if custom_lm.encode(l).size(0) > custom_lm.max_positions - 2:
            l = " ".join(l.split()[: custom_lm.max_positions - 2])
        tokens = custom_lm.encode(l).unsqueeze(0)
        print("Tokens:", tokens)
        print("Decoded Tokens:", custom_lm.decode(tokens))
        logprobs, extra = custom_lm.models[0](tokens)
        print("Log Probs Shape:", logprobs.shape)
        top_k_ids = logprobs[0, -2, :].argsort(descending=True)[: args.top_k].tolist()
        print("Top K token IDs:", top_k_ids)
        decoded_words = [custom_lm.decode([w]) for w in top_k_ids]
        print("Top K tokens:", decoded_words)
        top_k_data.append(
            {
                "context": l,
                "top_k_ids": top_k_ids,
                "top_k_decoded": decoded_words,
                "k": list(range(1, args.top_k + 1)),
            }
        )

    df = pd.DataFrame(top_k_data)
    df = df.explode(["top_k_ids", "top_k_decoded", "k"])
    df.to_csv(args.top_k_out_file, index=False)

