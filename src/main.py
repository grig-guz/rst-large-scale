import argparse
import os

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader

from data_helper import DataHelper
from eval.evaluation import Evaluator
from models.classifiers import NeuralClassifier
from models.parser import NeuralRstParser
from features.rst_dataset import RstDataset, BucketBatchSampler
from utils.constants import *

# To turn off long sequence tokenization warnings
import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true',
                        help='whether to extract feature templates, action maps and relation maps')
    parser.add_argument('--train', action='store_true',
                        help='whether to train new models')
    parser.add_argument('--eval', action='store_true',
                        help='whether to do evaluation')
    parser.add_argument('--train_dir', help='train data directory')
    parser.add_argument('--cuda_id', help='id of the gpu', default="0")
    parser.add_argument('--eval_dir', help='eval data directory')
    parser.add_argument('--dataset_type', help='which dataset to train/test on (RST/Instr/MEGA-DT)')
    parser.add_argument('--roberta_only', action='store_true', help='whether to run ablation only using roberta')
    parser.add_argument('--org_feats_only', action='store_true', help='whether to run ablation only using org feats')
    parser.add_argument('--finetune_megadt', help='if specified, will start training from the given checkpoint name of MEGA-DT pretrained model', default=None)
    parser.add_argument('--use_parseval', action='store_true', help='whether to use original Parseval (or RST-Parseval)')
    parser.add_argument('--model_name', help='Name of the model')
    parser.add_argument('--untrained_roberta', action='store_true', help='use roberta without pretraining')
    parser.add_argument('--epoch_start', help='Which epoch to start from', type=int, default=1)

    return parser.parse_args()

def collate_trees(trees, config):
    all_docs, all_action_seqs = [], []
    for doc, action_seq in trees:
        all_docs.append(doc)
        all_action_seqs.append(action_seq)
    return all_docs, all_action_seqs

def train_model(data_helper, helper_path, config):
    
    data_helper.load_data_helper(helper_path)
    config[DEVICE] = torch.cuda.current_device()
    action_fvs = data_helper.feats_list
    action_labels = data_helper.action_seqs_numeric
    if config[DATASET_TYPE] == MEGA_DT:
        _, action_fvs, _, action_labels = train_test_split(action_fvs, action_labels, test_size=52630, random_state=1)
        t_train, t_dev, l_train, l_dev = train_test_split(action_fvs, action_labels, test_size=0.05, random_state=1)
    else:
        t_train, t_dev, l_train, l_dev = train_test_split(action_fvs, action_labels, test_size=0.1, random_state=1)
    train_data = RstDataset(t_train, l_train, data_helper, is_train=True)
    dev_data = RstDataset(t_dev, l_dev, data_helper, is_train=False)
    bucket_batch_sampler = BucketBatchSampler(l_train, 
                                              config[BATCH_SIZE])
    train_loader = DataLoader(train_data, 
                               batch_sampler=bucket_batch_sampler,
                               batch_size=1, 
                               shuffle=False,
                               collate_fn=lambda x: collate_trees(x, config), 
                               drop_last=False)
    print("Train data:", len(train_data), "Dev data:", len(dev_data))
    dev_loader = DataLoader(dev_data, collate_fn=lambda x: x[0])
    clf = NeuralClassifier(data_helper, config)
    clf.to("cuda")
    rst_parser = NeuralRstParser(clf, config, save_dir=None)
    rst_parser.train_classifier(train_loader, dev_loader)

if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]= args.cuda_id

    config = {
        ORG_FEATS: True,
        HIDDEN_DIM: 512,
        BATCH_SIZE: 5,
        LR: 0.001,
        EPOCH_START: args.epoch_start,
        MODEL_NAME: args.model_name,
        DEVICE: torch.cuda.current_device(),
        KEEP_BOUNDARIES: False,
        ROBERTA_ONLY: args.roberta_only,
        ORG_FEATS_ONLY: args.org_feats_only,
        FINETUNE_MEGADT: args.finetune_megadt,
        UNTRAINED_ROBERTA: args.untrained_roberta,
        NUM_EPOCHS: 20
    }
    
    data_helper = DataHelper()
    
    config[DATASET_TYPE] = args.dataset_type
    assert config[DATASET_TYPE] in [RST, INSTR, MEGA_DT]
    assert not (config[ROBERTA_ONLY] and config[ORG_FEATS_ONLY])
    helper_path = os.path.join('../data/', "data_helper_" + config[DATASET_TYPE] + ".bin")
    model_name = args.model_name
    
    if args.prepare:
        data_helper.create_data_helper(args.train_dir, config)
        data_helper.save_data_helper(helper_path)
    if args.train:
        if not os.path.isdir('../data/model'):
            os.mkdir('../data/model')
        if args.model_name == None:
            raise Exception("Please provide model name")
        train_model(data_helper, helper_path, config)
    if args.eval:
        data_helper.load_data_helper(helper_path)
        clf = NeuralClassifier(data_helper, config)
        parser = NeuralRstParser(clf, config, save_dir=None)
        parser.load(os.path.join('../data/model/', args.model_name + "_" + str(config[EPOCH_START])))
        clf.eval()
        clf.to("cuda")
        with torch.no_grad():
            evaluator = Evaluator(parser, data_helper)
            evaluator.eval_parser(None, path=args.eval_dir, use_parseval=args.use_parseval)
