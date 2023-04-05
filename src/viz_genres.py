import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from tqdm import tqdm

if __name__ == "__main__":
    df = pd.read_excel("../data/coha/sources_coha.xlsx")
    df["decade"] = df.year.astype(int).apply(lambda x: x // 10 * 10)

    sns.histplot(
        data=df,
        x="decade",
        hue="genre",
        multiple="stack",
        bins=list(range(1810, 2020, 10)),
    )
    plt.savefig("../img/viz_genres", dpi=150, bbox_inches="tight")

    plt.clf()

    sns.histplot(
        data=df,
        x="decade",
        hue="genre",
        weights="# words",
        multiple="stack",
        bins=list(range(1810, 2020, 10)),
    )
    plt.savefig("../img/viz_genres_word_weighted", dpi=150, bbox_inches="tight")
