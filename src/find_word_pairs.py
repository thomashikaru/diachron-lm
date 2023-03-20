import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from fairseq.models.transformer_lm import TransformerLanguageModel
import seaborn as sns
import argparse
from tqdm import tqdm
import fastBPE
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv", default="../data/freq_analysis/delta_freqs.csv")
    parser.add_argument("--out_csv", default="../data/word_pairs/output.csv")
    parser.add_argument("--search_text", default="../data/coha/lm_data/2000/en.train")
    args = parser.parse_args()

    # read candidate list - words with high ratio of frequency in 2000s to frequency in first decade of use
    df = pd.read_csv(args.in_csv)
    words = df.word.iloc[-25:]

    # find sentences containing a candidate word
    data = []
    for word in words:
        rx = re.compile(rf".+\b{word}\b")
        with open(args.search_text) as f:
            for line in f:
                match = re.search(rx, line)
                if match:
                    data.append(
                        {
                            "word": word,
                            "sentence": match.group(0),
                            "context": match.group(0)[0 : -len(word) - 1],
                        }
                    )
    df = pd.DataFrame(data)
    df["wc"] = df.groupby("word")["word"].transform("count")
    df = df[df.wc >= 100]
    df = df.groupby("word").sample(n=100).reset_index()

    # calculate per-word surprisals for each sentence using modern data LM
    decade = 2000
    custom_lm = TransformerLanguageModel.from_pretrained(
        f"models/{decade}",
        data_name_or_path=f"data/coha/lm_data/{decade}/en-bin",
        checkpoint_file="checkpoint_best.pt",
    )
    custom_lm.eval()
    bpe = fastBPE.fastBPE(
        f"models/bpe_codes/30k/{decade}/en.codes",
        f"models/bpe_codes/30k/{decade}/en.vocab",
    )

    surprisals = []
    for sentence in tqdm(bpe.apply(df.sentence)):
        if custom_lm.encode(sentence).size(0) > custom_lm.max_positions - 2:
            sentence = " ".join(sentence.split()[: custom_lm.max_positions - 2])
        out = custom_lm.score(sentence, shorten_method="truncate")
        surprisals.append(-1 * out["positional_scores"][-1])
    df["surprisal"] = surprisals

    # identify contexts that are highly predictive of candidate word (surprisal < threshold)
    df = (
        df.sort_values("surprisal", ascending=False)
        .groupby("word")
        .head(10)
        .reset_index()
    )

    # using other historical LMs, find words that have high probability in these discovered contexts
    for decade in range(1830, 2000, 10):
        custom_lm = TransformerLanguageModel.from_pretrained(
            f"models/{decade}",
            data_name_or_path=f"data/coha/lm_data/{decade}/en-bin",
            checkpoint_file="checkpoint_best.pt",
        )
        custom_lm.eval()
        bpe = fastBPE.fastBPE(
            f"models/bpe_codes/30k/{decade}/en.codes",
            f"models/bpe_codes/30k/{decade}/en.vocab",
        )
        predictions = []
        for sentence in tqdm(bpe.apply(df.context)):
            if custom_lm.encode(sentence).size(0) > custom_lm.max_positions - 2:
                sentence = " ".join(sentence.split()[: custom_lm.max_positions - 2])
            tokens = custom_lm.encode(sentence).unsqueeze(0)
            logprobs, extra = custom_lm.models[0](tokens)
            top_k_ids = logprobs[0, -2, :].argsort(descending=True)[0].item()
            decoded = custom_lm.decode([top_k_ids])
            predictions.append(decoded)
        colname = f"top_pred_{decade}"
        df[colname] = predictions

    df.to_csv(args.out_csv, index=False)

