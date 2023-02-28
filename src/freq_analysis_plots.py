import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from tqdm import tqdm
import random
from scipy.stats import entropy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv", default="../data/freq_analysis/word_counts.csv")
    parser.add_argument(
        "--out_csv", default="../data/freq_analysis/low_entropy_words.csv"
    )
    parser.add_argument("--plot_prefix", default="../img/freq_by_decade_example")
    parser.add_argument("--num_output_words", type=int, default=100)
    parser.add_argument("--num_plots", type=int, default=10)
    parser.add_argument("--thresh", type=int, default=50)
    args = parser.parse_args()

    df = pd.read_csv(args.in_csv)

    # restrict to words that are in the X percentile by frequency in all time periods
    thresh = np.percentile(df.frequency, args.thresh)
    df = df[df.frequency > thresh]
    df["count"] = df.groupby("word")["decade"].transform("count")
    num_decades = df.decade.nunique()
    df = df.query("count == @num_decades")

    # calculate the entropy of the frequency distribution (higher entropy == more uniform)
    df["freq_entropy"] = df.groupby("word")["frequency"].transform(entropy)

    # get the lowest entropy words
    df = df.sort_values("freq_entropy").iloc[: args.num_output_words * num_decades]
    df.to_csv(args.out_csv)

    for i in range(args.num_plots):
        # sample 10 words
        words = list(df.word.unique())
        words = random.sample(words, 10)
        df_sub = df[df.word.isin(words)]

        plt.clf()
        sns.lineplot(data=df_sub, x="decade", y="frequency", hue="word")
        plt.savefig(f"{args.plot_prefix}_{i}", dpi=150, bbox_inches="tight")
