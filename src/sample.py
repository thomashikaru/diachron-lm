import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import glob
import utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--decade", type=int)
    parser.add_argument("--output_dir")
    parser.add_argument("--data_dir")
    parser.add_argument("--train_size", type=float, default=0.8)
    args = parser.parse_args()

    data = []

    for doc in utils.iter_coha_decade(args.decade, args.data_dir):
        sents = utils.process_document(doc)
        data.append(pd.DataFrame({"text": sents}))
    df = pd.concat(data)

    train_df, test_df = train_test_split(df, train_size=args.train_size)
    valid_df, test_df = train_test_split(test_df, train_size=0.5)

    train_df.to_csv(f"{args.output_dir}/en.train", header=False, index=False, sep="\t")
    valid_df.to_csv(f"{args.output_dir}/en.valid", header=False, index=False, sep="\t")
    test_df.to_csv(f"{args.output_dir}/en.test", header=False, index=False, sep="\t")

