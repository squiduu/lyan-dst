import pytorch_lightning as pl
import torch
import json
import itertools
from transformers.models.t5.tokenization_t5 import T5Tokenizer
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.optimization import get_linear_schedule_with_warmup

from evaluation_v2 import get_val_metrics, get_test_metrics


class LoseyDST(pl.LightningModule):
    def __init__(self, args, tokenizer: T5Tokenizer, backbone: T5ForConditionalGeneration):
        super().__init__()

        self.args = args
        self.tokenizer = tokenizer
        self.backbone = backbone

    def training_step(self, batch, batch_idx):
        self.backbone.train()
        outputs = self.backbone.forward(
            input_ids=batch["encoder_input"], attention_mask=batch["attention_mask"], labels=batch["decoder_output"]
        )

        self.log(
            name="train_loss",
            value=outputs.loss,
            batch_size=self.args.batch_size_per_gpu * self.args.gpus * self.args.accumulate_grad_batches,
        )

        return outputs.loss

    def validation_step(self, batch, batch_idx):
        self.backbone.eval()
        val_ids = self.backbone.generate(
            inputs=batch["encoder_input"],
            max_length=15,
            min_length=0,
            early_stopping=True,
            num_beams=self.args.num_beams,
        )

        return {"val_ids": val_ids, "val_labels": batch["labels"], "val_input_seq": batch["input_seq"]}

    def validation_epoch_end(self, outputs):
        outputs = {k: list(itertools.chain(*[o[k] for o in outputs])) for k in outputs[0]}
        val_tokens = [
            self.tokenizer.decode(token_ids=_val_tokens, clean_up_tokenization_spaces=False)
            for _val_tokens in outputs["val_ids"]
        ]

        val_slot_acc = get_val_metrics(
            val_gen=val_tokens, val_labels=outputs["val_labels"], val_input_seq=outputs["val_input_seq"]
        )

        self.log(name="val_slot_acc", value=val_slot_acc)

    def test_step(self, batch, batch_idx):
        self.backbone.eval()
        pred_ids = self.backbone.generate(
            inputs=batch["encoder_input"],
            max_length=15,
            min_length=0,
            early_stopping=True,
            num_beams=self.args.num_beams,
        )

        return {"pred_labels": pred_ids}

    def test_epoch_end(self, outputs):
        outputs = {k: list(itertools.chain(*[o[k] for o in outputs])) for k in outputs[0]}
        pred_labels = [
            self.tokenizer.decode(token_ids=_pred_label, clean_up_tokenization_spaces=False)
            for _pred_label in outputs["pred_labels"]
        ]

        with open(
            f"{self.args.output_dir}/mwoz{self.args.data_ver}/infr_{self.args.exp_no}.json", mode="w", encoding="utf-8"
        ) as infr_fp:
            json.dump(obj=pred_labels, fp=infr_fp, indent=4)
        print(f" ----- Save the inference result: {infr_fp.name} ----- ")

        # show metrics
        get_test_metrics(self.args)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {"params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.args.lr)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": get_linear_schedule_with_warmup(
                    optimizer, self.args.warmup_steps, self.args.num_training_steps
                ),
                "interval": "step",
            },
        }
