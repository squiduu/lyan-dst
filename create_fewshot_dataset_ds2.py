import json
import random
import argparse
from typing import List, Dict

EXT_DOMS = ["police", "hospital"]


def extract_fewshot_dataset(dialogs: List[Dict], EXT_DOMS: List[str], total_dom_cnts: dict, args) -> List[Dict]:
    dom_cnts = {k: 0 for k, _ in total_dom_cnts.items()}
    new_ds = []

    for _ in range(1000):
        i = random.randint(0, 7888)

        if len(dialogs[i]["domains"]) == 1:
            if (
                dialogs[i]["domains"][0] not in EXT_DOMS
                and dom_cnts[dialogs[i]["domains"][0]] < total_dom_cnts[dialogs[i]["domains"][0]] * args.fewshot
            ):
                new_ds.append(dialogs[i])
                dom_cnts[dialogs[i]["domains"][0]] += 1
        else:
            if (
                tuple(dialogs[i]["domains"]) not in EXT_DOMS
                and dom_cnts[tuple(dialogs[i]["domains"])] < total_dom_cnts[tuple(dialogs[i]["domains"])] * args.fewshot
            ):
                new_ds.append(dialogs[i])
                dom_cnts[tuple(dialogs[i]["domains"])] += 1

        if sum(dom_cnts.values()) >= sum(total_dom_cnts.values()) * args.fewshot:
            break

    return new_ds


def create_fewshot_dataset(args):
    total_dom_cnts = {}

    with open(args.orig_data_path) as f:
        dialogs = json.load(f)

    for i in range(len(dialogs)):
        if dialogs[i]["domains"][0] not in EXT_DOMS:
            if len(dialogs[i]["domains"]) == 1:
                if dialogs[i]["domains"][0] not in total_dom_cnts.keys():
                    total_dom_cnts[dialogs[i]["domains"][0]] = 1
                else:
                    total_dom_cnts[dialogs[i]["domains"][0]] += 1
            elif len(dialogs[i]["domains"]) > 1:
                if tuple(dialogs[i]["domains"]) not in total_dom_cnts.keys():
                    total_dom_cnts[tuple(dialogs[i]["domains"])] = 1
                else:
                    total_dom_cnts[tuple(dialogs[i]["domains"])] += 1

    new_ds = extract_fewshot_dataset(dialogs=dialogs, EXT_DOMS=EXT_DOMS, total_dom_cnts=total_dom_cnts, args=args)
    check_duplication = set(dialogs["dial_id"] for dialogs in new_ds)
    if len(check_duplication) != len(new_ds):
        new_ds = extract_fewshot_dataset(dialogs=dialogs, EXT_DOMS=EXT_DOMS, total_dom_cnts=total_dom_cnts)

    with open(f"./data/mwoz{args.data_ver}_train_1pct_{args.set_no}.json", mode="w", encoding="utf-8") as new_f:
        json.dump(obj=new_ds, fp=new_f, indent=4)

    print(f" ----- Extract {len(new_ds)} dialogues from MultiWOZ {args.data_ver} dataset. ----- ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--fewshot", default=0.01, type=float, required=False)
    parser.add_argument("--data_ver", default="21", type=str, required=False)
    parser.add_argument("--set_no", default="003", type=str, required=False)
    parser.add_argument(
        "--orig_data_path", default="../ds2_tj/ds2/data/mwoz21/train_dials.json", type=str, required=False
    )
    args = parser.parse_args()

    create_fewshot_dataset(args)
