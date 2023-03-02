import sys
import torch
from fairseq.models.transformer_lm import TransformerLanguageModel
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import fastBPE


def make_plot(surprisals, tokens, plot_dir, imgname):
    df = pd.DataFrame({"surprisal": surprisals, "token": tokens})
    df.surprisal = -1 * df.surprisal
    sns.axes_style("whitegrid")
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.lineplot(data=df, ax=ax)
    ax.set_xticks(range(len(df)), labels=list(df.token), ha="left", rotation=-45)
    plt.savefig(os.path.join(plot_dir, imgname), dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir")
    parser.add_argument("--data_dir")
    parser.add_argument("--test_file")
    parser.add_argument("--out_file")
    parser.add_argument("--plot_dir")
    parser.add_argument("--emb_out_file")
    parser.add_argument("--codes_path")
    parser.add_argument("--vocab_path")
    args = parser.parse_args()

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
    lines = bpe.apply(lines)

    # get per-token surprisals
    lprobs, perps, tokens = [], [], []
    for item, l in enumerate(lines):
        print(l)
        if custom_lm.encode(l).size(0) > custom_lm.max_positions - 2:
            l = " ".join(l.split()[: custom_lm.max_positions - 2])
        out = custom_lm.score(l, shorten_method="truncate")
        perps.append(out["positional_scores"].mean().neg().exp().item())
        lprobs.append(out["positional_scores"])
        print(out["positional_scores"])
        tokens.append([custom_lm.tgt_dict[i] for i in out["tokens"]])
        print(tokens[-1])
        make_plot(lprobs[-1], tokens[-1], args.plot_dir, f"sanity_check_plot_{item}")
    torch.save([lprobs, tokens], args.out_file)

    # get hidden representations
    all_embeddings = []
    for l in lines:
        if custom_lm.encode(l).size(0) > custom_lm.max_positions - 2:
            l = " ".join(l.split()[: custom_lm.max_positions - 2])
        tokens = custom_lm.encode(l).unsqueeze(0)
        print("Tokens:", tokens)
        print("Decoded Tokens:", custom_lm.decode(tokens))
        x, extra = custom_lm.models[0](tokens)
        print("Log Probs Shape:", x.shape)
        extracted_features = extra["inner_states"]
        all_embeddings.append(extracted_features)
        print("Final Hidden Layer Shape:", extracted_features[-1].shape)
    torch.save(all_embeddings, args.emb_out_file)

