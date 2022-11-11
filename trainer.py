import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.t5.tokenization_t5 import T5Tokenizer
import os

from modeling import LoseyDST
from config import get_args
from custom_dataloader import prepare_data, EQV_SLTS


def trainer(args):
    print(vars(args))
    # set wandb logger
    wandb_logger = WandbLogger(name=args.exp_no, save_dir=args.output_dir, project=args.project, version=args.exp_no)

    # set seed
    seed_everything(args.seed)

    # get pre-trained tokenizer for model
    print(f" ----- Get pre-trained tokenizer: {args.model_name_or_path} ----- ")
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    added_special_tokens = tokenizer.add_special_tokens(
        {
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
        }
    )
    # get pre-trained backbone network
    backbone = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    backbone.resize_token_embeddings(new_num_tokens=tokenizer.vocab_size + added_special_tokens)

    # get dataset
    print(" ----- Get dataloaders ----- ")
    dataloaders = prepare_data(tokenizer=tokenizer, EQV_SLTS=EQV_SLTS, args=args)

    # set max training steps
    if args.max_steps > 0:
        args.num_training_steps = args.max_steps
        args.max_epochs = args.max_steps // len(dataloaders["train"]) + 1
    else:
        args.num_training_steps = int(len(dataloaders["train"]) * args.max_epochs / args.gpus)

    # set warm-up steps
    if args.warmup_steps < 0:
        args.warmup_steps = int(len(dataloaders["train"]) / args.gpus * args.max_epochs * 0.2)
    # record warm-up steps
    print(f" ----- Warm-up steps: {args.warmup_steps} ----- ")

    if args.ckpt_path:
        print(f" ----- Get checkpoint from: {args.ckpt_path} ----- ")
        # get dst model from ckpt
        dst_model = LoseyDST.load_from_checkpoint(
            checkpoint_path=args.ckpt_path, args=args, tokenizer=tokenizer, backbone=backbone
        )
    else:
        print(f" ----- Get pre-trained model: {args.model_name_or_path} ----- ")
        # get dst model for fine tuning
        dst_model = LoseyDST(args=args, tokenizer=tokenizer, backbone=backbone)

    # model checkpoint callback
    if not args.do_test_only:
        es_callback = EarlyStopping(monitor="val_slot_acc", patience=args.patience, verbose=True, mode="max")
        ckpt_callback = ModelCheckpoint(
            filename="{epoch:02d}-{val_slot_acc:.2f}", save_top_k=1, monitor="val_slot_acc", mode="max"
        )
        callbacks = [es_callback, ckpt_callback]
    else:
        callbacks = None

    # set trainer
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        gradient_clip_val=args.gradient_clip_val,
        gpus=args.gpus,
        enable_progress_bar=True,
        accumulate_grad_batches=args.accumulate_grad_batches,
        max_epochs=args.max_epochs,
        strategy=args.strategy,
    )

    if not args.do_test_only:
        # start to train
        print(" ----- Start training ----- ")
        trainer.fit(model=dst_model, train_dataloaders=dataloaders["train"], val_dataloaders=dataloaders["dev"])

    if not args.do_train_only:
        print(" ----- Start testing ----- ")
        args.ckpt_path = os.path.join(
            f"{args.output_dir}/{args.exp_no}/{args.exp_no}/checkpoints",
            os.listdir(f"{args.output_dir}/{args.exp_no}/{args.exp_no}/checkpoints")[0],
        )
        print(f" ----- Get checkpoint from: {args.ckpt_path} ----- ")
        # get dst model from ckpt
        dst_model = LoseyDST.load_from_checkpoint(
            checkpoint_path=args.ckpt_path, args=args, tokenizer=tokenizer, backbone=backbone
        )
        trainer.test(model=dst_model, dataloaders=dataloaders["test"])


if __name__ == "__main__":
    args = get_args()
    trainer(args)
