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
import string

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_csv",
        default="/home/thclark/diachron-lm/data/freq_analysis/delta_freqs.csv",
    )
    parser.add_argument(
        "--freq_csv",
        default="/home/thclark/diachron-lm/data/freq_analysis/word_counts.csv",
    )
    parser.add_argument("--freq_thresh", default=1e-6)
    parser.add_argument(
        "--out_csv", default="/home/thclark/diachron-lm/data/word_pairs/output.csv"
    )
    parser.add_argument(
        "--search_text",
        default="/home/thclark/diachron-lm/data/coha/lm_data/2000/en.test",
    )
    parser.add_argument("--models_dir", default="/home/thclark/diachron-lm/models")
    parser.add_argument("--data_dir", default="/home/thclark/diachron-lm/data")
    parser.add_argument(
        "--candidate_count",
        type=int,
        default=200,
        help="number of diachronically usage-increasing words to consider",
    )
    parser.add_argument(
        "--shortlist_count",
        type=int,
        default=50,
        help="number of shortlist sentences (containing candidate words) to consider",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=5,
        help="number of top_n predictive contexts to save per candidate word, from the shortlist",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="number of top_k completions to save per context",
    )
    args = parser.parse_args()

    # read candidate list - words with high ratio of frequency in 2000s to frequency in first decade of use
    df = pd.read_csv(args.in_csv)
    high_freq_ratio_words = df.word.iloc[-args.candidate_count :]
    mid_freq_ratio_words = df.query(
        "delta_freq <= 1.25 & delta_freq >= 0.75"
    ).word.sample(n=args.candidate_count)
    low_freq_ratio_words = df.word.iloc[: args.candidate_count]
    words = pd.concat(
        [high_freq_ratio_words, mid_freq_ratio_words, low_freq_ratio_words]
    ).astype(str)
    categories = (
        ["increasing"] * len(high_freq_ratio_words)
        + ["flat"] * len(mid_freq_ratio_words)
        + ["decreasing"] * len(low_freq_ratio_words)
    )

    # find sentences containing a candidate word
    data = []

    # read sentences to be searched
    sentences = pd.read_csv(
        args.search_text, sep="\t", names=["sentence"], dtype={"sentence": str}
    )

    # apply filters to sentences (remove sentences with weird punctuation, low-frequency words, etc.)
    freqs = pd.read_csv(args.freq_csv)
    decade_count = freqs.decade.nunique()
    freqs = freqs.groupby("word").agg({"decade": list, "frequency": list}).reset_index()
    freqs = freqs[freqs.decade.apply(len) == decade_count]
    freqs = freqs[freqs.frequency.apply(lambda x: all(y > args.freq_thresh for y in x))]
    freq_words = set(freqs.word)
    print("Number of words in high-freq list:", len(set(freq_words)))
    sentences = sentences[
        sentences.sentence.apply(
            lambda x: sum(y not in freq_words for y in x.split()[:-1]) <= 3
        )
    ].reset_index(drop=True)

    # for each candidate word extract sentences containing that word
    for word, category in zip(words, categories):
        # exclude words containing punctuation
        if any(p in word for p in string.punctuation):
            continue
        rx = re.compile(rf"(.+\b{word}\b)")
        df_sub = (
            sentences.sentence.str.extract(rx)
            .copy()
            .dropna()
            .rename(columns={0: "sentence"})
        )
        df_sub["context"] = df_sub.sentence.apply(lambda x: x[0 : -len(word) - 1])
        df_sub["word"] = word
        df_sub["category"] = category
        try:
            df_sub = df_sub[df_sub.sentence.apply(lambda x: x.split().index(word) >= 5)]
        except ValueError as err:
            print(
                err,
                "\nThis should not happen because `word` should be guaranteed to be in the sentence.",
            )
        data.append(df_sub)
    df = pd.concat(data)
    df["wc"] = df.groupby("word")["word"].transform("count")
    df = df[df.wc >= args.shortlist_count]
    df = df.groupby("word").sample(n=args.shortlist_count).reset_index()

    # calculate per-word surprisals for each sentence using modern data LM
    decade = 2000
    custom_lm = TransformerLanguageModel.from_pretrained(
        f"{args.models_dir}/{decade}",
        data_name_or_path=f"{args.data_dir}/coha/lm_data/{decade}/en-bin",
        checkpoint_file="checkpoint_best.pt",
    )
    custom_lm.eval()
    bpe = fastBPE.fastBPE(
        f"{args.models_dir}/bpe_codes/30k/{decade}/en.codes",
        f"{args.models_dir}/bpe_codes/30k/{decade}/en.vocab",
    )

    surprisals = []
    for sentence in tqdm(bpe.apply(df.sentence)):
        if custom_lm.encode(sentence).size(0) > custom_lm.max_positions - 2:
            sentence = " ".join(sentence.split()[: custom_lm.max_positions - 2])
        out = custom_lm.score(sentence, shorten_method="truncate")
        surprisals.append(-1 * out["positional_scores"][-1])
    df["surprisal"] = surprisals

    # identify contexts that are highly predictive of candidate word
    df = (
        df.sort_values("surprisal", ascending=True)
        .groupby("word")
        .head(args.top_n)
        .reset_index()
        .sort_values("word")
    )

    # using other historical LMs, find words that have high probability in these discovered contexts
    explode_colnames = []
    for decade in range(1830, 2000, 10):
        custom_lm = TransformerLanguageModel.from_pretrained(
            f"{args.models_dir}/{decade}",
            data_name_or_path=f"{args.data_dir}/coha/lm_data/{decade}/en-bin",
            checkpoint_file="checkpoint_best.pt",
        )
        custom_lm.eval()
        bpe = fastBPE.fastBPE(
            f"{args.models_dir}/bpe_codes/30k/{decade}/en.codes",
            f"{args.models_dir}/bpe_codes/30k/{decade}/en.vocab",
        )
        predictions = []
        for sentence in tqdm(bpe.apply(df.context)):
            if custom_lm.encode(sentence).size(0) > custom_lm.max_positions - 2:
                sentence = " ".join(sentence.split()[: custom_lm.max_positions - 2])
            tokens = custom_lm.encode(sentence).unsqueeze(0)
            logprobs, extra = custom_lm.models[0](tokens)
            top_k_ids = (
                logprobs[0, -2, :].argsort(descending=True)[: args.top_k].tolist()
            )
            decoded = [custom_lm.decode([w]) for w in top_k_ids]
            predictions.append(decoded)
        colname = f"top_pred_{decade}"
        explode_colnames.append(colname)
        df[colname] = predictions

    ranks = [list(range(1, args.top_k + 1))] * len(df)
    df["rank"] = ranks
    explode_colnames.append("rank")

    df = df.explode(explode_colnames)
    df.to_csv(args.out_csv, index=False)

