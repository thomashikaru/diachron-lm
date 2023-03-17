import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from tqdm import tqdm
import random
from scipy.stats import entropy


def delta_freq(df):
    df = df.sort_values("decade")
    df_new = pd.DataFrame(
        {
            "word": [df.word.iloc[0]],
            "delta_freq": df.frequency.iloc[-1] / df.frequency.iloc[0],
            "first_use": df.decade.min(),
        }
    )
    return df_new


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv", default="../data/freq_analysis/word_counts.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.in_csv)

    keep_words = set(df.query("decade == 2000 & frequency > 1e-5").word)
    df = df[df.word.isin(keep_words)]

    df = df.groupby("word").apply(delta_freq)

    df = df.sort_values("delta_freq")
    df.to_csv("../data/freq_analysis/delta_freqs.csv", index=False)

