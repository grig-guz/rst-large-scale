import torch
from utils.constants import *


class ActionFeatureGenerator(object):
    
    def __init__(self, stack, queue, action_hist, doc, data_helper, config):
        """ Initialization of features generator

        :type stack: list
        :param stack: list of Node instance

        :type queue: list
        :param queue: list of Node instance

        :type doc: Doc instance
        :param doc:
        """
        # -------------------------------------
        self.action_hist = action_hist
        self.doc = doc
        self.stack = stack
        self.queue = queue
        self.data_helper = data_helper
        self.config = config
        # Stack
        if len(stack) >= 2:
            self.top1span, self.top2span, self.top3span = stack[-1], stack[-2], None
        elif len(stack) == 1:
            self.top1span, self.top2span, self.top3span = stack[-1], None, None
        else:
            self.top1span, self.top2span, self.top3span = None, None, None
        # Queue
        if len(queue) > 0:
            self.firstspan = queue[0]
        else:
            self.firstspan = None
        # Doc length wrt EDUs
        self.doclen = len(self.doc.edu_dict)

        
    def gen_features(self):
        """ Queue EDU + 3 subtrees from the top of the stack
        """
        feat_list = []
        neural_feats = []
        # Textual organization features
        if self.config[ORG_FEATS]:        
            for feat in self.organizational_features():
                feat_list.append(feat)
                
        if (self.firstspan is not None):
            neural_feats.append(("QueueEDUs1", self.firstspan.edu_span))
        if (self.top1span is not None):
            neural_feats.append(("StackEDUs1", self.top1span.edu_span))
        if (self.top2span is not None):
            neural_feats.append(("StackEDUs2", self.top2span.edu_span))
                
        return feat_list, neural_feats
    
    
    def organizational_features(self):
        trueTensor, falseTensor = torch.cuda.LongTensor([1]), torch.cuda.LongTensor([0])        
        # ---------------------------------------
        # Whether within same sentence and paragraph
        # Span 1 and 2
        if self.top1span is not None and self.top2span is not None:
            text1, text2 = self.top1span.text, self.top2span.text
            if self.doc.token_dict[text2[-1]].sidx == self.doc.token_dict[text1[0]].sidx:
                yield (TOP12_STACK, SENT_CONTINUE, trueTensor)
            else:
                yield (TOP12_STACK, SENT_CONTINUE, falseTensor)
            if self.doc.token_dict[text2[-1]].pidx == self.doc.token_dict[text1[0]].pidx:
                yield (TOP12_STACK, PARA_CONTINUE, trueTensor)
            else:
                yield (TOP12_STACK, PARA_CONTINUE, falseTensor)
        else:
            yield (TOP12_STACK, NOT_PRESENT)

        # Span 1 and top span
        # First word from span 1, last word from span 3
        if self.top1span is not None and self.firstspan is not None:
            text1, text3 = self.top1span.text, self.firstspan.text
            if self.doc.token_dict[text1[-1]].sidx == self.doc.token_dict[text3[0]].sidx:
                yield (STACK_QUEUE, SENT_CONTINUE, trueTensor)
            else:
                yield (STACK_QUEUE, SENT_CONTINUE, falseTensor)
            if self.doc.token_dict[text1[-1]].pidx == self.doc.token_dict[text3[0]].pidx:
                yield (STACK_QUEUE, PARA_CONTINUE, trueTensor)
            else:
                yield (STACK_QUEUE, PARA_CONTINUE, falseTensor)
        else:
            yield (STACK_QUEUE, NOT_PRESENT)

        # # Last word from span 1, first word from span 2
        top12_stack_same_sent, top12_stack_same_para = False, False
        if self.top1span is not None and self.top2span is not None:
            text1, text2 = self.top1span.text, self.top2span.text
            if self.doc.token_dict[text1[-1]].sidx == self.doc.token_dict[text2[0]].sidx:
                top12_stack_same_sent = True
                yield (TOP12_STACK, SAME_SENT, trueTensor)
            else:
                yield (TOP12_STACK, SAME_SENT, falseTensor)
            if self.doc.token_dict[text1[-1]].pidx == self.doc.token_dict[text2[0]].pidx:
                top12_stack_same_para = True
                yield (TOP12_STACK, SAME_PARA, trueTensor)
            else:
                yield (TOP12_STACK, SAME_PARA, falseTensor)
        else:
             yield (TOP12_STACK, NOT_PRESENT)

        # # Span 1 and top span
        # # First word from span 1, last word from span 3
        stack_queue_same_sent, stack_queue_same_para = False, False
        if self.top1span is not None and self.firstspan is not None:
            text1, text3 = self.top1span.text, self.firstspan.text
            if self.doc.token_dict[text1[0]].sidx == self.doc.token_dict[text3[-1]].sidx:
                stack_queue_same_sent = True
                yield (STACK_QUEUE, SAME_SENT, trueTensor)
            else:
                yield (STACK_QUEUE, SAME_SENT, falseTensor)
            if self.doc.token_dict[text1[0]].pidx == self.doc.token_dict[text3[-1]].pidx:
                stack_queue_same_para = True
                yield (STACK_QUEUE, SAME_PARA, trueTensor)
            else:
                yield (STACK_QUEUE, SAME_PARA, falseTensor)
        else:
             yield (STACK_QUEUE, NOT_PRESENT)
                
        
        if top12_stack_same_sent and stack_queue_same_sent:
            yield (TOP12_STACK_QUEUE, SAME_SENT, trueTensor)
        else:
            yield (TOP12_STACK_QUEUE, SAME_SENT, falseTensor)
            
        if top12_stack_same_para and stack_queue_same_para:
            yield (TOP12_STACK_QUEUE, SAME_PARA, trueTensor)
        else:
            yield (TOP12_STACK_QUEUE, SAME_PARA, falseTensor)
        
        # ---------------------------------------
        # whether span is the start or end of sentence, paragraph or document
        if self.top1span is not None:
            text = self.top1span.text
            if text[0] - 1 < 0 or self.doc.token_dict[text[0] - 1].sidx != self.doc.token_dict[text[0]].sidx:
                yield (TOP_1, SENT_START, trueTensor)
            else:
                yield (TOP_1, SENT_START, falseTensor)
                
            if text[-1] + 1 >= len(self.doc.token_dict) or self.doc.token_dict[text[-1] + 1].sidx != \
                    self.doc.token_dict[text[-1]].sidx:
                yield (TOP_1, SENT_END, trueTensor)
            else:
                yield (TOP_1, SENT_END, falseTensor)
                
            if text[0] - 1 < 0 or self.doc.token_dict[text[0] - 1].pidx != self.doc.token_dict[text[0]].pidx:
                yield (TOP_1, PARA_START, trueTensor)
            else:
                yield (TOP_1, PARA_START, falseTensor)
                
            if text[-1] + 1 >= len(self.doc.token_dict) or self.doc.token_dict[text[-1] + 1].pidx != \
                    self.doc.token_dict[text[-1]].pidx:
                yield (TOP_1, PARA_END, trueTensor)
            else:
                yield (TOP_1, PARA_END, falseTensor)
                
            if text[0] - 1 < 0:
                yield (TOP_1, DOC_START, trueTensor)
            else:
                yield (TOP_1, DOC_START, falseTensor)
                
            if text[-1] + 1 >= len(self.doc.token_dict):
                yield (TOP_1, DOC_END, trueTensor)
            else:
                yield (TOP_1, DOC_END, falseTensor)
                
        else:
            yield (TOP_1, NOT_PRESENT)
            
        if self.top2span is not None:
            text = self.top2span.text
            if text[0] - 1 < 0 or self.doc.token_dict[text[0] - 1].sidx != self.doc.token_dict[text[0]].sidx:
                yield (TOP_2, SENT_START, trueTensor)
            else:
                yield (TOP_2, SENT_START, falseTensor)
                
            if text[-1] + 1 >= len(self.doc.token_dict) or self.doc.token_dict[text[-1] + 1].sidx != \
                    self.doc.token_dict[text[-1]].sidx:
                yield (TOP_2, SENT_END, trueTensor)
            else:
                yield (TOP_2, SENT_END, falseTensor)
                
            if text[0] - 1 < 0 or self.doc.token_dict[text[0] - 1].pidx != self.doc.token_dict[text[0]].pidx:
                yield (TOP_2, PARA_START, trueTensor)
            else:
                yield (TOP_2, PARA_START, falseTensor)
                
            if text[-1] + 1 >= len(self.doc.token_dict) or self.doc.token_dict[text[-1] + 1].pidx != \
                    self.doc.token_dict[text[-1]].pidx:
                yield (TOP_2, PARA_END, trueTensor)
            else:
                yield (TOP_2, PARA_END, falseTensor)
                
            if text[0] - 1 < 0:
                yield (TOP_2, DOC_START, trueTensor)
            else:
                yield (TOP_2, DOC_START, falseTensor)
            if text[-1] + 1 >= len(self.doc.token_dict):
                yield (TOP_2, DOC_END, trueTensor)
            else:
                yield (TOP_2, DOC_END, falseTensor)
        else:
            yield (TOP_2, NOT_PRESENT)

            
        if self.firstspan is not None:
            text = self.firstspan.text
            if text[0] - 1 < 0 or self.doc.token_dict[text[0] - 1].sidx != self.doc.token_dict[text[0]].sidx:
                yield (QUEUE_1, SENT_START, trueTensor)
            else:
                yield (QUEUE_1, SENT_START, falseTensor)
            if text[-1] + 1 >= len(self.doc.token_dict) or self.doc.token_dict[text[-1] + 1].sidx != \
                    self.doc.token_dict[text[-1]].sidx:
                yield (QUEUE_1, SENT_END, trueTensor)
            else:
                yield (QUEUE_1, SENT_END, falseTensor)
            if text[0] - 1 < 0 or self.doc.token_dict[text[0] - 1].pidx != self.doc.token_dict[text[0]].pidx:
                yield (QUEUE_1, PARA_START, trueTensor)
            else:
                yield (QUEUE_1, PARA_START, falseTensor)
            if text[-1] + 1 >= len(self.doc.token_dict) or self.doc.token_dict[text[-1] + 1].pidx != \
                    self.doc.token_dict[text[-1]].pidx:
                yield (QUEUE_1, PARA_END, trueTensor)
            else:
                yield (QUEUE_1, PARA_END, falseTensor)
            if text[0] - 1 < 0:
                yield (QUEUE_1, DOC_START, trueTensor)
            else:
                yield (QUEUE_1, DOC_START, falseTensor)
            if text[-1] + 1 >= len(self.doc.token_dict):
                yield (QUEUE_1, DOC_END, trueTensor)
            else:
                yield (QUEUE_1, DOC_END, falseTensor)
        else:
            yield (QUEUE_1, NOT_PRESENT)
        