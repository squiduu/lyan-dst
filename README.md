# ryan-dst
Scripts of "Learn from Your Ancestor: Boosting Few-Shot Dialogue State Tracking with Pseudo Labels from Previous Models."

## Installation
This repository is available in Ubuntu 20.04 LTS, and it is not tested in other OS.
```
git clone https://github.com/squiduu/ryan-dst.git
cd ryan-dst

conda create -n ryandst python=3.7.10
conda activate ryandst

pip install -r requirements.txt
```

## Pre-processing
Extract 1% of dataset randomly
```
python create_fewshot_dataset_ds2.py --fewshot 0.01 --data_ver 21 --set_no003
```

## Pseudo labeling
```
python combine_pseudo_labels_ds2.py
```

## Fine-tuning
Default dataset is MultiWOZ2.1 for the `.sh` file. You can revise it with another versions of MultiWOZ datasets.
```
sh trainer.sh
```

## Inference
Revise `--ckpt_path` argument to your own `.ckpt` file.
```
sh inference.sh
```
