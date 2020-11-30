import os
import torch
from eval.metrics import Metrics
from models.tree import RstTree
from features.rst_dataset import *
from utils.document import Doc
from utils.constants import *


class Evaluator(object):
    def __init__(self, parser, data_helper):
        print('Load parsing models ...')
        self.parser = parser
        self.data_helper = data_helper

    @staticmethod
    def writebrackets(fname, brackets):
        """ Write the bracketing results into file"""
        with open(fname, 'w') as fout:
            for item in brackets:
                fout.write(str(item) + '\n')

    def eval_parser(self, dev_data=None, path='./examples', use_parseval=False):
        """ Test the parsing performance"""
        # Evaluation
        met = Metrics(use_parseval, levels=['span', 'nuclearity'])
        # ----------------------------------------
        # Read all files from the given path
        if dev_data is None:
            eval_list = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith('.merge')]
        else:
            eval_list = dev_data
        pred_forms = []
        gold_forms = []
        total_cost = 0
        for eval_instance in eval_list:
            # ----------------------------------------
            # Read *.merge file
            doc = Doc()
            if dev_data is not None:
                gold_rst = eval_instance
                doc = eval_instance.doc
            else:
                doc.read_from_fmerge(eval_instance)
                eval_instance = eval_instance.replace('.dis', '.merge')
                fdis = eval_instance.replace('.merge', '.dis')
                gold_rst = RstTree(fdis, eval_instance)
                gold_rst.build()
                
            _, gold_action_seq = gold_rst.generate_action_samples()
            gold_action_seq = list(map(lambda x: self.data_helper.action_map[x], gold_action_seq))
            pred_rst, cost = self.parser.sr_parse([doc], 
                                                  [torch.cuda.LongTensor(gold_action_seq)], 
                                                  None, 
                                                  is_train=False)
            total_cost += cost
            pred_rst = pred_rst[0]
            # ----------------------------------------
            # Evaluate with gold RST tree
            met.eval(gold_rst, pred_rst)
            
        print("Total cost: ", total_cost)
        met.report()