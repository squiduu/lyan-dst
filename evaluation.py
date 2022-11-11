import argparse
import json
import re


TIME_SLT = ["train arrive by", "train leave at", "taxi arrive at", "taxi leave at", "restaurant book time"]


def fix_time_format(unfixed: str):
    if re.match(r"\d{2}:\d{2}", unfixed) or unfixed in ["none", "dontcare"]:
        fixed = unfixed
    elif re.match(r"\d{1}:\d{2} \w{2}", unfixed):
        if unfixed[-2:] == "am":
            fixed = re.sub(r"\d{1}:\d{2} \w{2}", "0" + unfixed[0] + ":" + unfixed[2:4], unfixed)
        elif unfixed[-2:] == "pm":
            fixed = re.sub(r"\d{1}:\d{2} \w{2}", str(int(unfixed[0]) + 12) + ":" + unfixed[2:4], unfixed)
    elif re.match(r"\d{1}:\d{2}\w{2}", unfixed):
        if unfixed[-2:] == "am":
            fixed = re.sub(r"\d{1}:\d{2}\w{2}", "0" + unfixed[0] + ":" + unfixed[2:4], unfixed)
        elif unfixed[-2:] == "pm":
            fixed = re.sub(r"\d{1}:\d{2}\w{2}", str(int(unfixed[0]) + 12) + ":" + unfixed[2:4], unfixed)
    elif re.match("\d:\d{2}", unfixed):
        fixed = re.sub("\d:\d{2}", "0" + unfixed, unfixed)
    elif re.match("\d{4}", unfixed):
        fixed = re.sub("\d{4}", unfixed[:2] + ":" + unfixed[2:], unfixed)
    elif re.match("\d{3}", unfixed):
        fixed = re.sub("\d{3}", "0" + unfixed[0] + ":" + unfixed[1:], unfixed)
    elif re.match("\d{1} \w{2}", unfixed):
        if unfixed[-2:] == "am":
            fixed = re.sub("\d{1} \w{2}", "0" + unfixed[0] + ":00", unfixed)
        elif unfixed[-2:] == "pm":
            fixed = re.sub("\d{1} \w{2}", str(int(unfixed[0]) + 12) + ":00", unfixed)
    elif re.match("\d{2} \w{2}", unfixed):
        if unfixed[-2:] == "am":
            fixed = re.sub("\d{2} \w{2}", unfixed[:2] + ":00", unfixed)
        elif unfixed[-2:] == "pm":
            fixed = re.sub("\d{2} \w{2}", str(int(unfixed[:2]) + 12) + ":00", unfixed)
    elif re.match("\d{2}\w{2}", unfixed):
        if unfixed[-2:] == "am":
            fixed = re.sub("\d{2}\w{2}", unfixed[:2] + ":00", unfixed)
        elif unfixed[-2:] == "pm":
            fixed = re.sub("\d{2}\w{2}", str(int(unfixed[:2]) + 12) + ":00", unfixed)
    elif re.match("\d{1}\w{2}", unfixed):
        if unfixed[-2:] == "am":
            fixed = re.sub("\d{1}\w{2}", "0" + unfixed[0] + ":00", unfixed)
        elif unfixed[-2:] == "pm":
            fixed = re.sub("\d{1}\w{2}", str(int(unfixed[0]) + 12) + ":00", unfixed)
    else:
        fixed = unfixed
    try:
        return fixed
    except:
        print(unfixed)


def get_metrics(args):
    metrics = {}
    with open(f"./data/mwoz{args.data_ver}/test_preprocessed.json", mode="r", encoding="utf-8") as gold_fp:
        gold_labels = json.load(gold_fp)
    with open(f"{args.output_dir}/mwoz{args.data_ver}/infr_{args.exp_no}.json", mode="r", encoding="utf-8") as infr_fp:
        pred_labels = json.load(infr_fp)

    # get average slot accuracy
    match = []
    if args.use_judgment:
        for i in range(len(pred_labels)):
            gold = str.split(gold_labels[i]["labels"], sep="<extra_id_1>")[-1].split("<extra_id_2>")[0].strip()
            pred = str.split(pred_labels[i], sep="<extra_id_1>")[-1].split("<extra_id_2>")[0].strip()
    else:
        for i in range(len(pred_labels)):
            gold = str.split(gold_labels[i]["labels"], sep="<extra_id_0>")[-1].split("<extra_id_1>")[0].strip()
            pred = str.split(pred_labels[i], sep="<extra_id_0>")[-1].split("<extra_id_1>")[0].strip()

            if str.split(gold_labels[i]["input_seq"], sep="<belief state>")[-1].split("=")[0].strip() in TIME_SLT:
                gold = fix_time_format(gold)
                pred = fix_time_format(pred)

            if gold.replace(" ", "") == pred.replace(" ", ""):
                match.append(1.0)
            else:
                match.append(0.0)

    metrics["slot_acc"] = round(sum(match) / len(match) * 100, ndigits=2)

    # get average joint goal accuracy
    matches = []
    if args.use_judgment:
        for i in range(0, len(pred_labels), args.num_slots):
            match = []
            for j in range(i, i + args.num_slots):
                gold = str.split(gold_labels[j]["labels"], sep="<extra_id_1>")[-1].split("<extra_id_2>")[0].strip()
                pred = str.split(pred_labels[j], sep="<extra_id_1>")[-1].split("<extra_id_2>")[0].strip()
    else:
        for i in range(0, len(pred_labels), args.num_slots):
            match = []
            for j in range(i, i + args.num_slots):
                gold = str.split(gold_labels[j]["labels"], sep="<extra_id_0>")[-1].split("<extra_id_1>")[0].strip()
                pred = str.split(pred_labels[j], sep="<extra_id_0>")[-1].split("<extra_id_1>")[0].strip()

                if str.split(gold_labels[j]["input_seq"], sep="<belief state>")[-1].split("=")[0].strip() in TIME_SLT:
                    gold = fix_time_format(gold)
                    pred = fix_time_format(pred)

                if gold.replace(" ", "") == pred.replace(" ", ""):
                    match.append(1.0)
                else:
                    match.append(0.0)

            if sum(match) == len(match):
                matches.append(1.0)
            else:
                matches.append(0.0)

    metrics["joint_acc"] = round(sum(matches) / len(matches) * 100, ndigits=2)

    # get joint goal accuracy per domains
    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_ver", default="21", type=str, required=False)
    parser.add_argument("--output_dir", default="./out", type=str, required=False)
    parser.add_argument("--num_slots", default=30, type=int, required=False)
    parser.add_argument("--exp_no", default="exp15t", type=str, required=False)
    parser.add_argument("--use_judgment", action="store_true")
    args = parser.parse_args()

    get_metrics(args)
