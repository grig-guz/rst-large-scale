import os

from features.extraction import ActionFeatureGenerator
from models.state import ParsingState
from models.tree import RstTree
import torch
import torch.nn as nn

from utils.constants import *
from utils.other import cleanup_load_dict

from torch.nn import CrossEntropyLoss
from models.optimizer import Optimizer
from eval.evaluation import Evaluator


class NeuralRstParser(object):
    
    def __init__(self, clf, config, save_dir=None):
        self.clf = clf
        self.config = config
        self.save_dir = save_dir
        weight = torch.cuda.FloatTensor([0.5, 0.5/3, 0.5/3, 0.5/3])
        self.loss = CrossEntropyLoss(weight=weight, reduction='sum')
        self.optim = None
        if self.config[FINETUNE_MEGADT] is not None:
            self.load("../data/model/" + self.config[FINETUNE_MEGADT])


    def save(self, model_dir, idx, epoch):
        """Save models
        """
        torch.save({'epoch': epoch,
                    'model_state_dict': self.clf.state_dict(), 
                    'optimizer_state_dict': self.optim.state_dict(),
                    'step': self.optim._step},
                    os.path.join(model_dir, idx))

        
    def load(self, model_dir):
        """ Load models
        """
        model_save = torch.load(model_dir)
        # In case an old model is loaded
        cleanup_load_dict(model_save)        
        self.clf.load_state_dict(model_save['model_state_dict'])
        if self.optim is not None:
            self.optim.load_state_dict(model_save['optimizer_state_dict'], model_save['step'])
            self.optim._step = model_save['step']
        self.clf.eval()

        
    def train_classifier(self, train_loader, dev_loader):
        """
        """
        self.optim = Optimizer(self.clf.parameters(), lr=self.config[LR])
        if self.config[EPOCH_START] != 1:
            self.load('../data/model/' + self.config[MODEL_NAME] + "_" + str(self.config[EPOCH_START]))

        for epoch in range(1, self.config[NUM_EPOCHS] + 1):
            cost_acc = 0
            self.clf.train()
            print("============ epoch: ", epoch, " ============")
            for i, data in enumerate(train_loader):
                docs, gold_actions = data
                cost_acc += self.sr_parse(docs, gold_actions, self.optim)[1]
                if (i % 50 == 0):
                    print("Cost on step ", i, "is ", cost_acc)
                    
            print("Total cost for epoch ", epoch, "is ", cost_acc)
            self.clf.eval()
            with torch.no_grad():
                eval = Evaluator(self, self.clf.data_helper)
                eval.eval_parser(dev_loader, path=None)
            self.save('../data/model/', 
                      self.config[MODEL_NAME] + "_" + str(epoch), epoch)
                        
                
    def sr_parse(self, docs, gold_actions, optim, is_train=True):
        
        confs = []
        action_hists = []
        for doc in docs:
            conf = ParsingState([], [], self.config)
            conf.init(doc)
            confs.append(conf)
            action_hists.append([])
        action_hist_nonnum = []
        iter = 0
        cost_acc = 0
        while not confs[0].end_parsing():
            if is_train:
                optim.zero_grad()
            action_feats, neural_feats, spans, curr_gold_actions = [], [], [], []
            for conf, doc in zip(confs, docs):
                stack, queue = conf.get_status()
                fg = ActionFeatureGenerator(stack, 
                                            queue, 
                                            action_hist_nonnum, 
                                            doc, 
                                            self.clf.data_helper, 
                                            self.config)
                action_feat, neural_feat = fg.gen_features()
                action_feats.append(action_feat)
                neural_feats.append(neural_feat)
            
            spans = self.clf.get_edus_bert_span(docs, neural_feats)
            action_probs = self.clf.decode_action_online(spans, action_feats)
            sorted_action_idx = torch.argsort(action_probs, descending=True)
            
            for i, conf in enumerate(confs):
                action_hists[i].append(action_probs[i])
                action_idx = 0
                curr_gold_actions.append(gold_actions[i][iter].unsqueeze(0))
                if is_train:
                    action = self.clf.xidx_action_map[int(gold_actions[i][iter])]
                else:
                    action = self.clf.xidx_action_map[int(sorted_action_idx[i, action_idx])]
                    while not conf.is_action_allowed(action, docs[i]):
                        action = self.clf.xidx_action_map[int(sorted_action_idx[i, action_idx])]
                        action_idx += 1
                conf.operate(action)
            cost = self.loss(action_probs, torch.cat(curr_gold_actions))
            cost_acc += cost.item()
            if is_train:
                cost.backward()
                nn.utils.clip_grad_norm_(self.clf.parameters(), 0.2)
                optim.step()
            iter += 1
            
        rst_trees = []
        for conf, doc in zip(confs, docs):
            tree = conf.get_parse_tree()
            rst_tree = RstTree()
            rst_tree.assign_tree(tree)
            rst_tree.assign_doc(doc)
            rst_tree.back_prop(tree, doc)
            rst_trees.append(rst_tree)
        return rst_trees, cost_acc
