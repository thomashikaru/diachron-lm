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
        "--out_csv", default="../data/freq_analysis/high_entropy_words.csv"
    )
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
    df = df.sort_values("freq_entropy").iloc[:200]
    df.to_csv(args.out_csv)

    # sample 10 words
    words = list(df.word.unique())
    words = random.sample(words, 10)
    df = df[df.word.isin(words)]

    sns.lineplot(data=df, x="decade", y="frequency", hue="word")
    plt.savefig("../img/freq_by_decade_example", dpi=150, bbox_inches="tight")
