import json
import argparse


def combine_pseudo_labels(args):
    with open(file=args.pseudo_path, mode="r", encoding="utf-8") as pseudo:
        pseudo_data = json.load(pseudo)

    with open(file=args.orig_path, mode="r", encoding="utf-8") as orig_train_set:
        orig_train_set = json.load(orig_train_set)

    data = []
    idx = 0
    for single_dialog in orig_train_set:
        turns = []
        for turn, pseudo in zip(single_dialog["turns"], pseudo_data[idx : idx + len(single_dialog["turns"])]):
            turn["state"]["pseudo_labels"] = pseudo["pseudo_labels"]
            turns.append(turn)
        idx += len(single_dialog["turns"])

        data.append({"dial_id": single_dialog["dial_id"], "domains": single_dialog["domains"], "turns": turns})

    with open(
        file=f"./mwoz{args.data_ver}_updated_{args.fewshot}_{args.set_no}.json", mode="w", encoding="utf-8"
    ) as new_fp:
        json.dump(obj=data, fp=new_fp, indent=4)

    print(f" ----- Get pseudo-combined dataset: {new_fp.name} ----- ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--pseudo_path", default="./out/mwoz21_1pct_00_pseudo.json", type=str, required=False)
    parser.add_argument("--orig_path", default="./data/mwoz21/dev_dials.json", type=str, required=False)
    args = parser.parse_args()

    combine_pseudo_labels(args)
