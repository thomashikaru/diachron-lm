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
    parser.add_argument("--plot_prefix", default="../img/word_pair_plot")
    parser.add_argument("--words")
    args = parser.parse_args()

    df = pd.read_csv(args.in_csv)

    words = args.words.split(",")
    df = df[df.word.isin(words)]
    word_str = "_".join(words)

    sns.lineplot(data=df, x="decade", y="frequency", hue="word")
    plt.savefig(f"{args.plot_prefix}_{word_str}", dpi=150, bbox_inches="tight")
