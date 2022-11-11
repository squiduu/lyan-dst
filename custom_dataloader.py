import json
from torch.utils.data import Dataset, DataLoader
from transformers.models.auto.tokenization_auto import AutoTokenizer
from typing import List, Dict, Tuple

EXP_DOMS = set(["hotel", "train", "restaurant", "attraction", "taxi"])
EQV_SLTS = {
    "hotel price range": "hotel-pricerange",
    "hotel type": "hotel-type",
    "hotel parking": "hotel-parking",
    "hotel book stay": "hotel-book stay",
    "hotel book day": "hotel-book day",
    "hotel book people": "hotel-book people",
    "hotel area": "hotel-area",
    "hotel stars": "hotel-stars",
    "hotel internet": "hotel-internet",
    "train destination": "train-destination",
    "train day": "train-day",
    "train departure": "train-departure",
    "train arrive by": "train-arriveby",
    "train book people": "train-book people",
    "train leave at": "train-leaveat",
    "attraction area": "attraction-area",
    "restaurant food": "restaurant-food",
    "restaurant price range": "restaurant-pricerange",
    "restaurant area": "restaurant-area",
    "attraction name": "attraction-name",
    "restaurant name": "restaurant-name",
    "attraction type": "attraction-type",
    "hotel name": "hotel-name",
    "taxi leave at": "taxi-leaveat",
    "taxi destination": "taxi-destination",
    "taxi departure": "taxi-departure",
    "restaurant book time": "restaurant-book time",
    "restaurant book day": "restaurant-book day",
    "restaurant book people": "restaurant-book people",
    "taxi arrive by": "taxi-arriveby",
    "hotel-pricerange": "hotel price range",
    "hotel-type": "hotel type",
    "hotel-parking": "hotel parking",
    "hotel-book stay": "hotel book stay",
    "hotel-book day": "hotel book day",
    "hotel-book people": "hotel book people",
    "hotel-area": "hotel area",
    "hotel-stars": "hotel stars",
    "hotel-internet": "hotel internet",
    "train-destination": "train destination",
    "train-day": "train day",
    "train-departure": "train departure",
    "train-arriveby": "train arrive by",
    "train-book people": "train book people",
    "train-leaveat": "train leave at",
    "attraction-area": "attraction area",
    "restaurant-food": "restaurant food",
    "restaurant-pricerange": "restaurant price range",
    "restaurant-area": "restaurant area",
    "attraction-name": "attraction name",
    "restaurant-name": "restaurant name",
    "attraction-type": "attraction type",
    "hotel-name": "hotel name",
    "taxi-leaveat": "taxi leave at",
    "taxi-destination": "taxi destination",
    "taxi-departure": "taxi departure",
    "restaurant-book time": "restaurant book time",
    "restaurant-book day": "restaurant book day",
    "restaurant-book people": "restaurant book people",
    "taxi-arriveby": "taxi arrive by",
}
CND_VAL = {
    "hotel price range": "none, cheap, moderate, expensive, and dontcare.",
    "hotel type": "none, hotel, guest house, guesthouse, and dontcare.",
    "hotel parking": "none, yes, no, and dontcare.",
    "hotel book stay": "1, 2, 3, 4, 5, 6, 7, 8, none, and dontcare.",
    "hotel book day": "none, dontcare, monday, tuesday, wednesday, thursday, friday, saturday, and sunday.",
    "hotel book people": "1, 2, 3, 4, 5, 6, 7, 8, none, and dontcare.",
    "hotel area": "all possible values.",
    "hotel stars": "none, 0, 1, 2, 3, 4, 5, and dontcare.",
    "hotel internet": "none, yes, no, dontcare.",
    "train destination": "all possible values.",
    "train day": "none, dontcare, monday, tuesday, wednesday, thursday, friday, saturday, and sunday.",
    "train departure": "all possible values.",
    "train arrive by": "all possible values.",
    "train leave at": "all possible values.",
    "train book people": "all possible values.",
    "attraction area": "all possible values.",
    "restaurant food": "all possible values.",
    "restaurant price range": "none, cheap, moderate, expensive, and dontcare.",
    "restaurant area": "none, dontcare, centre, south, north, east, and west.",
    "attraction name": "all possible values.",
    "restaurant name": "all possible values.",
    "attraction type": "none, museum, entertainment, college, nightclub, swimming pool, multiple sports, architecture, cinema, boat, theatre, concert hall, park, dontcare, local site, hotspot, church, and special.",
    "hotel name": "all possible values.",
    "taxi leave at": "all possible values.",
    "taxi destination": "all possible values.",
    "taxi departure": "all possible values.",
    "restaurant book time": "all possible values.",
    "restaurant book day": "none, monday, tuesday, wednesday, thursday, dontcare, friday, saturday, and sunday.",
    "restaurant book people": "all possible values.",
    "taxi arrive by": "all possible values.",
}


class CustomizedDataset(Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data, args):
        """Reads source and target sequences from txt files."""
        self.data = data
        self.args = args

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """Returns one data pair - source and target."""
        return self.data[index]


def normalize_ontology(ontology: Dict[str, List[str]]) -> Dict[str, List[str]]:
    keys = [k for k in ontology]
    for k in keys:
        for i in range(len(ontology[k])):
            ontology[k][i] = ontology[k][i].replace("do n't care", "dontcare")
            ontology[k][i] = ontology[k][i].replace("'s", " s")

        ontology[k.lower() if ("book" not in k) else k.lower()] = ontology.pop(k)

    return ontology


def get_slot_information(ontology: Dict[str, List[str]]) -> List:
    ontology_domains = dict([(k, v) for k, v in ontology.items() if k.split("-")[0] in EXP_DOMS])
    ALL_SLTS = [k.lower() if ("book" not in k) else k.lower() for k in ontology_domains.keys()]
    ALL_SLTS = [str.replace(slot, "-", " ") for slot in ALL_SLTS]

    return ALL_SLTS


def get_key(input_dict: dict, input_value: str):
    for key, value in input_dict.items():
        if input_value == value:
            return str.replace(key, "-", " ")


def make_sequence(EQV_SLTS: dict, slot: str, gold_labels: dict, pseudo_labels: dict, args) -> Tuple[(str, str)]:
    if EQV_SLTS[slot] in gold_labels.keys():
        if gold_labels[EQV_SLTS[slot]] != "none":
            if EQV_SLTS[slot] in pseudo_labels.keys():
                if args.use_judgment:
                    judgment = "True" if gold_labels[EQV_SLTS[slot]] == pseudo_labels[EQV_SLTS[slot]] else "False"
                    prompt = (
                        "<belief state> "
                        + slot
                        + " = "
                        + pseudo_labels[EQV_SLTS[slot]]
                        + ", <extra_id_0>, "
                        + slot
                        + " = <extra_id_1>"
                    )
                    labels = (
                        "<extra_id_0> " + judgment + " <extra_id_1> " + gold_labels[EQV_SLTS[slot]] + " <extra_id_2>"
                    )
                else:
                    prompt = (
                        "<belief state> "
                        + slot
                        + " = "
                        + pseudo_labels[EQV_SLTS[slot]]
                        + ", </belief state> If it is wrong, find the correct one among "
                    )
                    labels = gold_labels[EQV_SLTS[slot]]
            else:
                if args.use_judgment:
                    prompt = "<belief state> " + slot + " = none, <extra_id_0>, " + slot + " = <extra_id_1>"
                    labels = "<extra_id_0> False <extra_id_1> " + gold_labels[EQV_SLTS[slot]] + " <extra_id_2>"
                else:
                    prompt = (
                        "<belief state> "
                        + slot
                        + " = none. </belief state> If it is wrong, find the correct one among "
                    )
                    labels = gold_labels[EQV_SLTS[slot]]
        else:
            if EQV_SLTS[slot] in pseudo_labels.keys():
                if args.use_judgment:
                    judgment = "True" if pseudo_labels[EQV_SLTS[slot]] == "none" else "False"
                    prompt = (
                        "<belief state> "
                        + slot
                        + " = "
                        + pseudo_labels[EQV_SLTS[slot]]
                        + ", <extra_id_0>, "
                        + slot
                        + " = <extra_id_1>"
                    )
                    labels = "<extra_id_0> " + judgment + " <extra_id_1> none <extra_id_2>"
                else:
                    prompt = (
                        "<belief state> "
                        + slot
                        + " = "
                        + pseudo_labels[EQV_SLTS[slot]]
                        + ". </belief state> If it is wrong, find the correct one among "
                    )
                    labels = "none"
            else:
                if args.use_judgment:
                    prompt = "<belief state> " + slot + " = none, <extra_id_0>, " + slot + " = <extra_id_1>"
                    labels = "<extra_id_0> True <extra_id_1> none <extra_id_2>"
                else:
                    prompt = (
                        "<belief state> "
                        + slot
                        + " = none. </belief state> If it is wrong, find the correct one among "
                    )
                    labels = "none"
    else:
        if EQV_SLTS[slot] in pseudo_labels.keys():
            if args.use_judgment:
                prompt = (
                    "<belief state> "
                    + slot
                    + " = "
                    + pseudo_labels[EQV_SLTS[slot]]
                    + ", <extra_id_0>, "
                    + slot
                    + " = <extra_id_1>"
                )
                labels = "<extra_id_0> False <extra_id_1> none <extra_id_2>"
            else:
                prompt = (
                    "<belief state> "
                    + slot
                    + " = "
                    + pseudo_labels[EQV_SLTS[slot]]
                    + ". </belief state> If it is wrong, find the correct one among "
                )
                labels = "none"
        else:
            if args.use_judgment:
                prompt = "<belief state> " + slot + " = none, <extra_id_0>, " + slot + " = <extra_id_1>"
                labels = "<extra_id_0> True <extra_id_1> none <extra_id_2>"
            else:
                prompt = (
                    "<belief state> " + slot + " = none. </belief state> If it is wrong, find the correct one among "
                )
                labels = "none"

    return prompt, labels


def make_inversed_sequence(
    EQV_SLTS: dict, slot: str, gold_labels: dict, pseudo_labels: dict, args
) -> Tuple[(str, str)]:
    if EQV_SLTS[slot] in gold_labels.keys():
        if gold_labels[EQV_SLTS[slot]] != "none":
            if EQV_SLTS[slot] in pseudo_labels.keys():
                if args.use_judgment:
                    judgment = "True" if gold_labels[EQV_SLTS[slot]] == pseudo_labels[EQV_SLTS[slot]] else "False"
                    prompt = (
                        "<belief state> "
                        + slot
                        + " = "
                        + pseudo_labels[EQV_SLTS[slot]]
                        + ", <extra_id_0>, <extra_id_1> = "
                        + gold_labels[EQV_SLTS[slot]]
                    )
                    labels = (
                        "<extra_id_0> "
                        + judgment
                        + " <extra_id_1> "
                        + get_key(gold_labels, gold_labels[EQV_SLTS[slot]])
                        + " <extra_id_2>"
                    )
                else:
                    prompt = (
                        "<belief state> "
                        + slot
                        + " = "
                        + pseudo_labels[EQV_SLTS[slot]]
                        + ", </belief state> If it is wrong, find the correct one among "
                    )
                    labels = gold_labels[EQV_SLTS[slot]]
            else:
                if args.use_judgment:
                    prompt = (
                        "<belief state> "
                        + slot
                        + " = "
                        + gold_labels[EQV_SLTS[slot]]
                        + ", <extra_id_0>, <extra_id_1> = "
                        + gold_labels[EQV_SLTS[slot]]
                    )
                    labels = "<extra_id_0> True <extra_id_1> " + slot + " <extra_id_2>"
                else:
                    prompt = (
                        "<belief state> "
                        + slot
                        + " = none. </belief state> If it is wrong, find the correct one among "
                    )
                    labels = gold_labels[EQV_SLTS[slot]]
        else:
            if EQV_SLTS[slot] in pseudo_labels.keys():
                if args.use_judgment:
                    prompt = (
                        "<belief state> "
                        + slot
                        + " = "
                        + pseudo_labels[EQV_SLTS[slot]]
                        + ", <extra_id_0>, <extra_id_1> = none"
                    )
                    labels = "<extra_id_0> False <extra_id_1> " + slot + " <extra_id_2>"
                else:
                    prompt = (
                        "<belief state> "
                        + slot
                        + " = "
                        + pseudo_labels[EQV_SLTS[slot]]
                        + ". </belief state> If it is wrong, find the correct one among "
                    )
                    labels = "none"
            else:
                if args.use_judgment:
                    prompt = "<belief state> " + slot + " = none, <extra_id_0>, <extra_id_1> = none"
                    labels = "<extra_id_0> True <extra_id_1> " + slot + " <extra_id_2>"
                else:
                    prompt = (
                        "<belief state> "
                        + slot
                        + " = none. </belief state> If it is wrong, find the correct one among "
                    )
                    labels = "none"
    else:
        if EQV_SLTS[slot] in pseudo_labels.keys():
            if args.use_judgment:
                prompt = (
                    "<belief state> "
                    + slot
                    + " = "
                    + pseudo_labels[EQV_SLTS[slot]]
                    + ", <extra_id_0>, <extra_id_1> = none"
                )
                labels = "<extra_id_0> False <extra_id_1> " + slot + " <extra_id_2>"
            else:
                prompt = (
                    "<belief state> "
                    + slot
                    + " = "
                    + pseudo_labels[EQV_SLTS[slot]]
                    + ". </belief state> If it is wrong, find the correct one among "
                )
                labels = "none"
        else:
            if args.use_judgment:
                prompt = "<belief state> " + slot + " = none, <extra_id_0>, <extra_id_1> = none"
                labels = "<extra_id_0> True <extra_id_1> " + slot + " <extra_id_2>"
            else:
                prompt = (
                    "<belief state> " + slot + " = none. </belief state> If it is wrong, find the correct one among "
                )
                labels = "none"

    return prompt, labels


def read_data(data_path: str, ALL_SLTS: list, EQV_SLTS: dict, run_type: str, args) -> List[Dict]:
    print(f" ----- Read datasets from {data_path} ----- ")

    dataset = []
    with open(data_path, mode="r", encoding="utf-8") as f:
        total_dialogs = json.load(f)

        for single_dialog in total_dialogs:
            dialog_history = "<dialogue>"

            for _, turn in enumerate(single_dialog["turns"]):
                # accumulate dialogue utterances
                dialog_history += " <sys> " + turn["system"] + " </sys> <usr> " + turn["user"] + " </usr>"

                for slot in ALL_SLTS:
                    prompt, labels = make_sequence(
                        EQV_SLTS=EQV_SLTS,
                        slot=slot,
                        gold_labels=turn["state"]["slot_values"],
                        pseudo_labels=turn["state"]["pseudo_labels"],
                        args=args,
                    )
                    if args.use_judgment:
                        data_line = {
                            "input_seq": dialog_history + " </dialogue> " + prompt + " </belief state>",
                            "labels": labels,
                        }
                    else:
                        data_line = {"input_seq": dialog_history + " </dialogue> " + prompt, "labels": labels}
                    dataset.append(data_line)

                    if run_type in ["train", "dev"] and args.use_inverse_prompt:
                        if EQV_SLTS[slot] in dict.keys(turn["state"]["slot_values"]):
                            if turn["state"]["slot_values"][EQV_SLTS[slot]] != "none":
                                if EQV_SLTS[slot] in dict.keys(turn["state"]["pseudo_labels"]):
                                    prompt, labels = make_inversed_sequence(
                                        EQV_SLTS=EQV_SLTS,
                                        slot=slot,
                                        gold_labels=turn["state"]["slot_values"],
                                        pseudo_labels=turn["state"]["pseudo_labels"],
                                        args=args,
                                    )
                                    if args.use_judgment:
                                        data_line = {
                                            "input_seq": dialog_history + " </dialogue> " + prompt + " </belief state>",
                                            "labels": labels,
                                        }
                                    else:
                                        data_line = {
                                            "input_seq": dialog_history + " </dialogue> " + prompt,
                                            "labels": labels,
                                        }
                                    dataset.append(data_line)

    if run_type == "test":
        with open(file=f"./data/mwoz{args.data_ver}/test_preprocessed.json", mode="w", encoding="utf-8") as infr_fp:
            json.dump(obj=dataset, fp=infr_fp, indent=4)
        print(f" ----- Save preprocessed test file: {infr_fp.name} ----- ")

    return dataset


def customized_collate_fn(tokenizer: AutoTokenizer, args):
    def _collate(batch):
        batch_data = {}
        for key in batch[0]:
            batch_data[key] = [data_line[key] for data_line in batch]

        input_batch = tokenizer(
            batch_data["input_seq"],
            padding=True,
            return_tensors="pt",
            add_special_tokens={
                "additional_special_tokens": [
                    "<dialogue>",
                    "<sys>",
                    "</sys>",
                    "<usr>",
                    "</usr>",
                    "</dialogue>",
                    "<belief state>",
                    "</belief state>",
                ]
            },
            verbose=False,
            truncation=True,
            max_length=args.enc_max_seq_len * 2,
        )

        batch_data["encoder_input"] = input_batch["input_ids"][:, -args.enc_max_seq_len :]
        batch_data["attention_mask"] = input_batch["attention_mask"][:, -args.enc_max_seq_len :]
        batch_data["decoder_output"] = tokenizer(
            batch_data["labels"],
            padding=True,
            return_tensors="pt",
            return_attention_mask=False,
            truncation=True,
            max_length=20,
        ).input_ids

        return batch_data

    return _collate


def prepare_data(tokenizer: AutoTokenizer, EQV_SLTS: dict, args) -> Dict[str, DataLoader]:
    data_paths = {
        "train": args.train_path,
        "dev": args.dev_path,
        "test": args.test_path,
    }
    ontology = normalize_ontology(ontology=json.load(open(file=f"./data/mwoz{args.data_ver}/ontology.json", mode="r")))
    ALL_SLTS = get_slot_information(ontology=ontology)

    datasets = {
        run_type: CustomizedDataset(
            data=read_data(data_path=data_path, ALL_SLTS=ALL_SLTS, EQV_SLTS=EQV_SLTS, run_type=run_type, args=args),
            args=args,
        )
        for run_type, data_path in data_paths.items()
    }

    dataloaders = {
        run_type: DataLoader(
            dataset=dataset,
            batch_size=args.bsz_per_gpu,
            shuffle=True if run_type == "train" else False,
            num_workers=args.num_workers,
            collate_fn=customized_collate_fn(tokenizer=tokenizer, args=args),
            pin_memory=True,
        )
        for run_type, dataset in datasets.items()
    }

    return dataloaders
