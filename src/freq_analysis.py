import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import glob
from tqdm import tqdm
import utils
from collections import Counter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir")
    parser.add_argument("--decades")
    parser.add_argument("--out_csv", default="../data/freq_analysis/word_counts.csv")
    args = parser.parse_args()

    decades = utils.DECADES
    if args.decades is not None:
        decades = [int(x) for x in args.decades.split(",")]

    dfs = []

    for decade in decades:
        print(decade)
        decade_counts = Counter()
        for doc in utils.iter_coha_decade(decade, args.data_dir):
            sents = utils.process_document(doc)
            text = " ".join(sents)
            doc_counts = Counter(text.split())
            decade_counts.update(doc_counts)
        df = pd.DataFrame(decade_counts.most_common())
        print(df)
        df.columns = ["word", "frequency"]
        df.frequency = df.frequency / sum(decade_counts.values())
        df["decade"] = decade
        dfs.append(df)

    df = pd.concat(dfs)
    df.to_csv(args.out_csv, index=False)
