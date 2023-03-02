import sys
import torch
from fairseq.models.transformer_lm import TransformerLanguageModel
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir")
    parser.add_argument("--data_dir")
    parser.add_argument("--test_file")
    parser.add_argument("--out_file")
    parser.add_argument("--emb_out_file")
    args = parser.parse_args()

    custom_lm = TransformerLanguageModel.from_pretrained(
        args.checkpoint_dir,
        data_name_or_path=args.data_dir,
        checkpoint_file="checkpoint_best.pt",
    )
    custom_lm.eval()

    with open(args.test_file, "r") as f:
        lines = f.read().splitlines()

    count = 0
    lprobs, perps, tokens = [], [], []
    for l in lines:
        print(l)
        if custom_lm.encode(l).size(0) > custom_lm.max_positions - 2:
            l = " ".join(l.split()[: custom_lm.max_positions - 2])
        out = custom_lm.score(l, shorten_method="truncate")
        perps.append(out["positional_scores"].mean().neg().exp().item())
        lprobs.append(out["positional_scores"])
        tokens.append([custom_lm.tgt_dict[i] for i in out["tokens"]])
    torch.save([lprobs, tokens], args.out_file)

    all_embeddings = []
    # hidden representations
    for l in lines:
        if custom_lm.encode(l).size(0) > custom_lm.max_positions - 2:
            l = " ".join(l.split()[: custom_lm.max_positions - 2])
        tokens = custom_lm.encode(l).unsqueeze(0)
        x, extra = custom_lm.models[0](tokens)
        extracted_features = extra["inner_states"]
        all_embeddings.append(extracted_features)
        print(extracted_features.shape)

    torch.save(all_embeddings, args.emb_out_file)

