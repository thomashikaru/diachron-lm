import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from tqdm import tqdm
import glob
import torch
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_pattern")
    parser.add_argument("--reference_file")
    parser.add_argument("--out_file")
    parser.add_argument("--out_csv")
    parser.add_argument("--layer_num", type=int, default=-1)
    parser.add_argument("--token_pos", type=int, default=-1)
    parser.add_argument("--decade", type=int)
    args = parser.parse_args()

    embeddings = []
    df_data = []

    words = []

    # reference word
    with open(args.reference_file) as f:
        sentences = f.read().strip().split("\n")
        print(len(sentences))
    for sentence in sentences:
        words.append(sentence.split()[args.token_pos])

    # embeddings
    filenames = glob.glob(args.file_pattern)
    for filename in filenames:
        data = torch.load(filename)
        assert len(data) == len(words), f"{len(data)} {len(words)}"
        for word, item in zip(words, data):
            embedding = (
                item[args.layer_num][args.token_pos, :, :].detach().squeeze().numpy()
            )
            embeddings.append(embedding)
            df_data.append({"word": word, "embedding": embedding})

    df = pd.DataFrame(df_data)
    # df.to_csv(args.out_csv, index=False)

    X = np.asarray(embeddings, dtype="float64")
    print(X.shape, X.dtype)
    print(X)
    X_embedded = TSNE(n_components=2, init="random", perplexity=3).fit_transform(X)
    df["tsne_0"] = X_embedded[:, 0]
    df["tsne_1"] = X_embedded[:, 1]

    sns.scatterplot(data=df, x="tsne_0", y="tsne_1")

    for i in range(0, df.shape[0]):
        plt.text(
            df.tsne_0[i] + 0.2,
            df.tsne_1[i],
            df.word[i],
            horizontalalignment="left",
            size="medium",
            color="black",
        )
    plt.savefig(
        args.out_file + "_embeddings", dpi=180, bbox_inches="tight",
    )

    cos_sims = cosine_similarity(X)

    plt.clf()
    df_plot = pd.DataFrame(cos_sims, index=words, columns=words)
    fig = plt.subplots(figsize=(8, 8))
    ax = sns.heatmap(
        df_plot,
        cmap="plasma",
        xticklabels=True,
        yticklabels=True,
        annot=True,
        fmt=".3f",
        annot_kws={"size": 14},
    )
    plt.title("Cosine Similarity of Intensifiers in Context", fontsize=20)
    plt.xlabel("Intensifier", fontsize=16)
    plt.ylabel("Intensifier", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.xaxis.tick_top()
    plt.savefig(args.out_file + "_heatmap", dpi=180, bbox_inches="tight")

