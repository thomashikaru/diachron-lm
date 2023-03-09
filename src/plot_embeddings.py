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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_pattern_1")
    parser.add_argument("--file_pattern_2")
    parser.add_argument("--reference_file_1")
    parser.add_argument("--reference_file_2")
    parser.add_argument("--save_dir")
    parser.add_argument("--layer_num", type=int, default=-1)
    parser.add_argument("--decade", type=int)
    args = parser.parse_args()

    embeddings = []
    df_data = []

    for reference_file, file_pattern in zip(
        [args.reference_file_1, args.reference_file_2],
        [args.file_pattern_1, args.file_pattern_2],
    ):
        words = []

        # reference word
        with open(reference_file) as f:
            sentences = f.read().strip().split("\n")
            print(len(sentences))
        for sentence in sentences:
            words.append(sentence.split()[-1])

        # embeddings
        filenames = glob.glob(file_pattern)
        for filename in filenames:
            data = torch.load(filename)
            assert len(data) == len(words), f"{len(data)} {len(words)}"
            for word, item in zip(words, data):
                embedding = item[args.layer_num][-1, :, :].detach().squeeze().numpy()
                embeddings.append(embedding)
                df_data.append({"word": word, "embedding": embedding})
            # print(len(data))
            # print(len(data[0]))
            # print(data[0][0].shape)

    df = pd.DataFrame(df_data)
    print(df)

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
        os.path.join(args.save_dir, f"embeddings_{args.decade}"),
        dpi=150,
        bbox_inches="tight",
    )

