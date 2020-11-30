import os
import pickle

from models.tree import RstTree
import torch


class DataHelper(object):
    
    def __init__(self):
        self.action_map = {}
        self.feats_list = []
        self.action_seqs_numeric = []
        
    def create_data_helper(self, data_dir, config):
        # read train data
        for i, rst_tree in enumerate(self.read_rst_trees(data_dir=data_dir)):
            sample = rst_tree.generate_action_samples()
            self.feats_list.append(rst_tree.fdis)
            self.action_seqs_numeric.append(sample[1])
            self.action_map = {('Shift', None): 0, 
                               ('Reduce', 'NS'): 1, 
                               ('Reduce', 'NN'): 2, 
                               ('Reduce', 'SN'): 3}
            
            if i % 50 == 0 and i > 0:
                print("Processed ", i, " trees")

        for i, action_seq in enumerate(self.action_seqs_numeric):
            self.action_seqs_numeric[i] = list(map(lambda x: self.action_map[x], action_seq))
            
        
    def save_data_helper(self, fname):
        print('Save data helper...')
        data_info = {
            'action_map':          self.action_map,
            'feats_list':          self.feats_list,
            'action_seqs_numeric': self.action_seqs_numeric
        }
        
        with open(fname, 'wb') as fout:
            pickle.dump(data_info, fout)

    def load_data_helper(self, fname):
        print('Load data helper ...')
        with open(fname, 'rb') as fin:
            data_info = pickle.load(fin)
        self.action_map = data_info['action_map']
        self.feats_list = data_info['feats_list']
        self.action_seqs_numeric = data_info['action_seqs_numeric']
                    
    @staticmethod
    def read_rst_trees(data_dir):
        # Read RST tree file
        files = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.dis')]
        for i, fdis in enumerate(files):
            fmerge = fdis.replace('.dis', '.merge')
            if not os.path.isfile(fmerge):
                print(fmerge + " - Corresponding .fmerge file does not exist. Skipping the file.")
                continue
            rst_tree = RstTree(fdis, fmerge)
            rst_tree.build()
            if (rst_tree.get_parse().startswith(" ( EDU")):
                print("Skipping document - tree is corrupt.")
                continue
            yield rst_tree