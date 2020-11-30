# RST Discourse Parsing with Large-Scale Pretraining
This repository contains the code for our experiments [(see our paper)](https://arxiv.org/abs/2011.03203) on improving RST discourse parsing with pretraining on [MEGA-DT corpus](https://www.cs.ubc.ca/cs-research/lci/research-groups/natural-language-processing/mega_dt.html). Our code is built upon [this repository](https://github.com/yizhongw/StageDP).

### Preparing data
Place the contents of training and testing portions of RST-DT inside the data/data_dir folder. It should look like this:
 ```bash
data/data_dir/train_dir/*
data/data_dir/test_dir/*
src/
```

### Preprocessing
  This project relies on Stanford CoreNLP 3.9.2 toolkit to preprocess the data. You can download it from [here](https://stanfordnlp.github.io/CoreNLP/history.html) and put the file [run_corenlp.sh](./run_corenlp.sh) into the CoreNLP folder. Then use the following command to preprocess both the data in train_dir and in test_dir:
    
  ```
  python preprocess.py --data_dir DATA_DIR --corenlp_dir CORENLP_DIR
  ```
Preprocessing output for this step is already included in MEGA-DT corpus and so is not required for it.

Next, run the following to generate the shift-reduce training sequences:
  ```
  python main.py --prepare --train_dir TRAIN_DIR --dataset_type rst
  ```
where ```--dataset_type``` flag is among  ```rst/instr/mega_dt```.

#### Testing (Optional)
To make sure that everything is ok so far, run the testing script:
  ```
  python test.py
  ```

### Training
Specify the name for which to save your model under, and run as follows:
 ```
python main.py --train --model_name YOUR_MODEL_NAME --dataset-type TYPE --epoch_start 1
```
The ```epoch_start``` flag lets you resume the training from the specified epoch. Additional flags (such as for training ablated models) can be located inside ```src/main.py```. If you already have a model pretrained on MEGA-DT, add the flag ```--finetune_megadt MODELNAME_EPOCH``` to use this model as the initial checkpoint.
### Evaluation
Similar to above:
 ```
python main.py --eval --eval_dir ../data/data_dir/test_dir/ --dataset_type TYPE --model_name YOUR_MODEL_NAME --epoch_start BEST_EPOCH
 ```
Add the flag ```--use_parseval``` to get the micro precision results with respect to Parseval metric, as opposed to RST-Parseval. 
