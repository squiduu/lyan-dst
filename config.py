import argparse


def get_args():
    parser = argparse.ArgumentParser()

    # set required arguments
    parser.add_argument("--data_dir", default="./data/", type=str, required=True)
    parser.add_argument(
        "--train_path", default="./data/mwoz21/train_extended_1pct_seed47_set00.json", type=str, required=True
    )
    parser.add_argument("--dev_path", default="./data/mwoz21/dev_extended.json", type=str, required=True)
    parser.add_argument("--test_path", default="./data/mwoz21/test_extended.json", type=str, required=True)
    parser.add_argument("--model_name_or_path", default="t5-base", type=str, required=True)
    parser.add_argument("--output_dir", default="./out/", type=str, required=True)
    parser.add_argument("--strategy", default="ddp", type=str, required=True)
    parser.add_argument("--batch_size_per_gpu", default=1, type=int, required=True)
    parser.add_argument("--num_workers", default=4, type=int, required=True)
    parser.add_argument("--lr", default=2e-5, type=float, required=True)
    parser.add_argument("--accumulate_grad_batches", default=1, type=int, required=True)
    parser.add_argument("--weight_decay", default=0.01, type=float, required=True)
    parser.add_argument("--exp_no", default="mwoz21-1pct-set00", type=str, required=True)
    parser.add_argument("--project", default="ds2-seed47", type=str, required=True)
    parser.add_argument("--gradient_clip_val", default=1.0, type=float, required=True)
    parser.add_argument("--warmup_steps", default=-1, type=int, required=True)
    parser.add_argument("--seed", default=42, type=int, required=True)
    parser.add_argument("--max_steps", default=-1, type=int, required=True)
    parser.add_argument("--gpus", default=1, type=int, required=True)
    parser.add_argument("--max_epochs", default=100, type=int, required=True)
    parser.add_argument("--do_train_only", action="store_true")
    parser.add_argument("--do_test_only", action="store_true")
    parser.add_argument("--num_beams", default=1, type=int, required=True)
    parser.add_argument("--patience", default=10, type=int, required=True)
    parser.add_argument("--data_ver", default="21", type=str, required=True)
    parser.add_argument("--only_domain", type=str, choices=["hotel", "train", "restaurant", "attraction", "taxi"])
    parser.add_argument("--except_domain", type=str, choices=["hotel", "train", "restaurant", "attraction", "taxi"])
    parser.add_argument("--ckpt_path", type=str, required=False)
    parser.add_argument("--num_slots", default=30, type=int, required=True)
    parser.add_argument("--judgment_weight", default=1.5, type=float, required=False)
    parser.add_argument("--enc_max_seq_len", default=1024, type=int, required=False)
    parser.add_argument("--use_judgment", action="store_true")
    parser.add_argument("--use_inverse_prompt", action="store_true")

    args = parser.parse_args()

    return args
