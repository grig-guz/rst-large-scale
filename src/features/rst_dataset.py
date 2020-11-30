import torch
from torch.utils.data import Sampler, Dataset
from utils.document import Doc
from models.tree import RstTree
from random import shuffle
from collections import OrderedDict

class RstDataset(Dataset):

    def __init__(self, X, y, data_helper, is_train=True):
        self.X = X
        self.y = y
        self.data_helper = data_helper
        self.is_train = is_train

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """
            Returns:
                document_edus - packed list of tensors, where each element is tensor of
                                word embeddings for an EDU of that document
                sorted_idx    - output of torch.argsort, used to return EDUs to original
                                linear order
                y             - sequence of gold actions for a particular tree
                doc           - Document instance of an RstTree instance
        """
        doc = Doc()
        eval_instance = self.X[idx].replace('.dis', '.merge')
        doc.read_from_fmerge(eval_instance)
        
        if self.is_train:
            return doc, torch.cuda.LongTensor(self.y[idx])
        
        fdis = eval_instance.replace('.merge', '.dis')
        gold_rst = RstTree(fdis, eval_instance)
        gold_rst.build()
        return gold_rst     
        
class BucketBatchSampler(Sampler):
    # want inputs to be an array
    def __init__(self, inputs, batch_size):
        self.batch_size = batch_size
        ind_n_len = []
        for i, p in enumerate(inputs):
            ind_n_len.append((i, len(p)))
        self.ind_n_len = ind_n_len
        self.batch_list = self._generate_batch_map()
        self.num_batches = len(self.batch_list)

    def _generate_batch_map(self):
        # shuffle all of the indices first so they are put into buckets differently
        shuffle(self.ind_n_len)
        # Organize lengths, e.g., batch_map[10] = [30, 124, 203, ...] <= indices of sequences of length 10
        batch_map = OrderedDict()
        for idx, length in self.ind_n_len:
            if length not in batch_map:
                batch_map[length] = [idx]
            else:
                batch_map[length].append(idx)
        # Use batch_map to split indices into batches of equal size
        # e.g., for batch_size=3, batch_list = [[23,45,47], [49,50,62], [63,65,66], ...]
        batch_list = []
        for length, indices in batch_map.items():
            for group in [indices[i:(i + self.batch_size)] for i in range(0, len(indices), self.batch_size)]:
                batch_list.append(group)
        return batch_list

    def batch_count(self):
        return self.num_batches

    def __len__(self):
        return len(self.ind_n_len)

    def __iter__(self):
        self.batch_list = self._generate_batch_map()
        # shuffle all the batches so they arent ordered by bucket size
        shuffle(self.batch_list)
        for i in self.batch_list:
            yield i