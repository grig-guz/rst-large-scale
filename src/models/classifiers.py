from utils.other import reverse_dict
from utils.constants import *

import torch
from transformers import RobertaTokenizer, RobertaModel
import torch.nn as nn

class NeuralClassifier(nn.Module):
    
    def __init__(self, data_helper, config):
        super(NeuralClassifier, self).__init__()    
        self.data_helper = data_helper
        self.xidx_action_map = reverse_dict(data_helper.action_map)
        self.config = config
        self.init_embeddings()
        self.tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')
        self.bert = RobertaModel.from_pretrained('distilroberta-base')

        if self.config[UNTRAINED_ROBERTA]:
            # Initialize new model with the same config as distilroberta
            self.bert = RobertaModel(self.bert.config)
            
        self.out_classifier = nn.Sequential(
            nn.Linear(self.get_classifier_dim(), self.config[HIDDEN_DIM]),
            nn.GELU(),
            nn.Linear(self.config[HIDDEN_DIM], 4)
        )
        
            
    def init_embeddings(self):
        # Wang etal features, encode binary(True/False) features
        self.t12_same_sent_feats = nn.Embedding(2, 10)    
        self.t12_same_para_feats = nn.Embedding(2, 10)
        self.st_q_same_sent_feats = nn.Embedding(2, 10)    
        self.st_q_same_para_feats = nn.Embedding(2, 10)
        self.t12_q_same_sent_feats = nn.Embedding(2, 10)    
        self.t12_q_same_para_feats = nn.Embedding(2, 10)
        
        self.top1_sent_start_feats = nn.Embedding(2, 10)
        self.top1_para_start_feats = nn.Embedding(2, 10)
        self.top1_doc_start_feats = nn.Embedding(2, 10)
        self.top1_sent_end_feats = nn.Embedding(2, 10)
        self.top1_para_end_feats = nn.Embedding(2, 10)
        self.top1_doc_end_feats = nn.Embedding(2, 10)
        self.top2_sent_start_feats = nn.Embedding(2, 10)
        self.top2_para_start_feats = nn.Embedding(2, 10)
        self.top2_doc_start_feats = nn.Embedding(2, 10)
        self.top2_sent_end_feats = nn.Embedding(2, 10)
        self.top2_para_end_feats = nn.Embedding(2, 10)
        self.top2_doc_end_feats = nn.Embedding(2, 10)

        self.queue1_sent_start_feats = nn.Embedding(2, 10)
        self.queue1_para_start_feats = nn.Embedding(2, 10)
        self.queue1_doc_start_feats = nn.Embedding(2, 10)
        self.queue1_sent_end_feats = nn.Embedding(2, 10)
        self.queue1_para_end_feats = nn.Embedding(2, 10)
        self.queue1_doc_end_feats = nn.Embedding(2, 10)
        
        self.pad_60 = torch.zeros((1, 60), device=self.config[DEVICE])
        self.pad_20 = torch.zeros((1, 20), device=self.config[DEVICE])        
    
    def decode_action_online(self, edu_embeds, action_feats):
        clf_feats = [[] for x in range(edu_embeds.shape[0])]
        edu_embeds = edu_embeds[:, 0, :]
        for i, node in enumerate(edu_embeds):
            if self.config[ROBERTA_ONLY]:
                clf_feats[i] = node.unsqueeze(0)
            elif self.config[ORG_FEATS_ONLY]:
                clf_feats[i] = self.add_action_feats(torch.cuda.FloatTensor([]), action_feats[i]).unsqueeze(0)
            else:
                clf_feats[i] = self.add_action_feats(node, action_feats[i]).unsqueeze(0)
        clf_feats = torch.cat(clf_feats, dim=0)
        return self.out_classifier(clf_feats)
        
    def get_classifier_dim(self):
        classifier_dim = 0
        if not self.config[ORG_FEATS_ONLY]:
            classifier_dim = 768
        if self.config[ORG_FEATS] and not self.config[ROBERTA_ONLY]:
            classifier_dim += 280
        return classifier_dim    
    
    def get_edus_bert_span(self, docs, spans): 
        # Fixed lengths, start and end
        span_embeds, masks = [], []
        total_spans = []
        left_bdry, right_bdry, strlen = 0, 1, 512
        batch_ids = []
        for i, (doc, span) in enumerate(zip(docs, spans)):
            first_span, second_span, third_span = [1]*28, [1]*240, [1]*240
            # First queue element
            if len(span) >= 1:
                first_span = self.extract_span_tokens(span, doc, first_span, 0)
            # First stack element
            if len(span) >= 2:
                second_span = self.extract_span_tokens(span, doc, second_span, 1)
            # Second stack element
            if len(span) >= 3:
                third_span = self.extract_span_tokens(span, doc, third_span, 2)
            total_span = [0] + third_span + [2] + second_span + [2] + first_span + [2]
            total_spans.append(torch.cuda.LongTensor(total_span).unsqueeze(0))
            
        span_edus = torch.cat(total_spans)
        span_masks = span_edus != 1
        node_word_tensors = self.bert(span_edus, attention_mask=span_masks.byte())[0]
        
        return node_word_tensors
    
    def extract_span_tokens(self, span, doc, span_tokens, span_num):
        left_bdry, right_bdry = 0, 1
        if span_num > 0:
            span_length = 120
        else:
            span_length = 14
            
        full_span_tokens = " ".join(doc.doc_edus[span[span_num][1][left_bdry] - 1:span[span_num][1][right_bdry]])
        full_span_tokens = self.tokenizer.encode(full_span_tokens, 
                                           add_special_tokens=False)
        if len(full_span_tokens) > span_length * 2:
            span_tokens[:span_length] = full_span_tokens[:span_length]
            span_tokens[span_length:] = full_span_tokens[-span_length:]
        else:
            span_tokens[:len(full_span_tokens)] = full_span_tokens
        return span_tokens
            
    def add_action_feats(self, node_edus, action_feats):
        feat_vector = []

        for i, feature in enumerate(action_feats):
            if feature[0] == TOP12_STACK:
                if feature[1] == NOT_PRESENT:
                    feat_vector.append(self.pad_20)
                elif feature[1] in [SAME_SENT, SENT_CONTINUE]:
                    feat_vector.append(self.t12_same_sent_feats(feature[2]))
                elif feature[1] in [SAME_PARA, PARA_CONTINUE]:
                    feat_vector.append(self.t12_same_para_feats(feature[2]))
                else:
                    raise ValueError("Unrecognized feature: ", feature)
            elif feature[0] == STACK_QUEUE:
                if feature[1] == NOT_PRESENT:
                    feat_vector.append(self.pad_20)
                elif feature[1] in [SAME_SENT, SENT_CONTINUE]:
                    feat_vector.append(self.st_q_same_sent_feats(feature[2]))
                elif feature[1] in [SAME_PARA, PARA_CONTINUE]:
                    feat_vector.append(self.st_q_same_para_feats(feature[2]))
                else:
                    raise ValueError("Unrecognized feature: ", feature)                    
            elif feature[0] == TOP12_STACK_QUEUE:
                if feature[1] == SAME_SENT:
                    feat_vector.append(self.t12_q_same_sent_feats(feature[2]))
                elif feature[1] == SAME_PARA:
                    feat_vector.append(self.t12_q_same_para_feats(feature[2]))
                else:
                    raise ValueError("Unrecognized feature: ", feature)
            elif feature[0] == TOP_1:
                if feature[1] == NOT_PRESENT:
                    feat_vector.append(self.pad_60)
                elif feature[1] == SENT_START:
                    feat_vector.append(self.top1_sent_start_feats(feature[2]))
                elif feature[1] == SENT_END:
                    feat_vector.append(self.top1_sent_end_feats(feature[2]))
                elif feature[1] == PARA_START:
                    feat_vector.append(self.top1_para_start_feats(feature[2]))
                elif feature[1] == PARA_END:
                    feat_vector.append(self.top1_para_end_feats(feature[2]))
                elif feature[1] == DOC_START:
                    feat_vector.append(self.top1_doc_start_feats(feature[2]))
                elif feature[1] == DOC_END:
                    feat_vector.append(self.top1_doc_end_feats(feature[2]))             
                else:
                    raise ValueError("Unrecognized feature: ", feature)
            elif feature[0] == TOP_2:
                if feature[1] == NOT_PRESENT:
                    feat_vector.append(self.pad_60)
                elif feature[1] == SENT_START:
                    feat_vector.append(self.top2_sent_start_feats(feature[2]))
                elif feature[1] == SENT_END:
                    feat_vector.append(self.top2_sent_end_feats(feature[2]))
                elif feature[1] == PARA_START:
                    feat_vector.append(self.top2_para_start_feats(feature[2]))
                elif feature[1] == PARA_END:
                    feat_vector.append(self.top2_para_end_feats(feature[2]))
                elif feature[1] == DOC_START:
                    feat_vector.append(self.top2_doc_start_feats(feature[2]))
                elif feature[1] == DOC_END:
                    feat_vector.append(self.top2_doc_end_feats(feature[2]))             
                else:
                    raise ValueError("Unrecognized feature: ", feature)
            elif feature[0] == QUEUE_1:
                if feature[1] == NOT_PRESENT:
                    feat_vector.append(self.pad_60)
                elif feature[1] == SENT_START:
                    feat_vector.append(self.queue1_sent_start_feats(feature[2]))
                elif feature[1] == SENT_END:
                    feat_vector.append(self.queue1_sent_end_feats(feature[2]))
                elif feature[1] == PARA_START:
                    feat_vector.append(self.queue1_para_start_feats(feature[2]))
                elif feature[1] == PARA_END:
                    feat_vector.append(self.queue1_para_end_feats(feature[2]))
                elif feature[1] == DOC_START:
                    feat_vector.append(self.queue1_doc_start_feats(feature[2]))
                elif feature[1] == DOC_END:
                    feat_vector.append(self.queue1_doc_end_feats(feature[2]))             
                else:
                    raise ValueError("Unrecognized feature: ", feature)
            else:
                raise ValueError("Unrecognized feature: ", feature)
                
        if feat_vector != []:
            feat_vector.append(node_edus.unsqueeze(0))
            node_edus = torch.cat(feat_vector, dim=1).squeeze(0)
            
        return node_edus