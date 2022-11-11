import json
import argparse


def get_extended_dataset(args):
    with open(
        file=f"./data/mwoz{args.data_ver}/train_{args.fewshot}_seed{args.seed}_set{args.set_no}_ratio30.json",
        mode="r",
        encoding="utf-8",
    ) as f1:
        f1 = json.load(f1)

    with open(
        file=f"./data/mwoz{args.data_ver}/train_{args.fewshot}_seed{args.seed}_set{args.set_no}_ratio50.json",
        mode="r",
        encoding="utf-8",
    ) as f2:
        f2 = json.load(f2)

    with open(
        file=f"./data/mwoz{args.data_ver}/train_{args.fewshot}_seed{args.seed}_set{args.set_no}_ratio80.json",
        mode="r",
        encoding="utf-8",
    ) as f3:
        f3 = json.load(f3)

    ext_data = []
    ext_data.extend(f1)
    ext_data.extend(f2)
    ext_data.extend(f3)

    with open(
        file=f"./data/mwoz{args.data_ver}/train_extended_{args.fewshot}_seed{args.seed}_set{args.set_no}.json",
        mode="w",
        encoding="utf-8",
    ) as ext_fp:
        json.dump(obj=ext_data, fp=ext_fp, indent=4)

    print(f" ----- Get extended dataset: {ext_fp.name} ----- ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_ver", default="21", type=str, required=False)
    parser.add_argument("--fewshot", default="1pct", type=str, required=False)
    parser.add_argument("--seed", default="47", type=str, required=False)
    parser.add_argument("--set_no", default="00", type=str, required=False)
    args = parser.parse_args()

    get_extended_dataset(args)
