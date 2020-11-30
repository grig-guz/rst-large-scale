
class ParseError(Exception):
    """ Exception for parsing
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class ActionError(Exception):
    """ Exception for illegal parsing action
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
    
def reverse_dict(dct):
    """ Reverse the {key:val} in dct to
        {val:key}
    """
    newmap = {}
    for (key, val) in dct.items():
        newmap[val] = key
    return newmap

def cleanup_load_dict(model_save):
    if 'root_embed' in model_save['model_state_dict']:
        del model_save['model_state_dict']['root_embed']
        del model_save['model_state_dict']['word_embeddings.weight']
        del model_save['model_state_dict']['pos_embeddings.weight']
        del model_save['model_state_dict']['dep_embeddings.weight']
        del model_save['model_state_dict']['positional_embeddings.pe']
        del model_save['model_state_dict']['operational_feats.weight']
        del model_save['model_state_dict']['t12_dep_type_feats.weight']
        del model_save['model_state_dict']['st_q_dep_type_feats.weight']
        del model_save['model_state_dict']['subtree_form_feats.weight']
        del model_save['model_state_dict']['edu_length_feats.weight']
        del model_save['model_state_dict']['sent_length_feats.weight']
        del model_save['model_state_dict']['edu_comp_feats.weight']
        del model_save['model_state_dict']['sent_comp_feats.weight']
        del model_save['model_state_dict']['stack_cat_feats.weight']
        del model_save['model_state_dict']['queue_cat_feats.weight']


class2rel = {
    'Attribution': ['attribution', 'attribution-e', 'attribution-n', 'attribution-negative'],
    'Background': ['background', 'background-e', 'circumstance', 'circumstance-e'],
    'Cause': ['cause', 'cause-result', 'result', 'result-e', 'consequence', 'consequence-n-e', 'consequence-n',
              'consequence-s-e', 'consequence-s'],
    'Comparison': ['comparison', 'comparison-e', 'preference', 'preference-e', 'analogy', 'analogy-e', 'proportion'],
    'Condition': ['condition', 'condition-e', 'hypothetical', 'contingency', 'otherwise'],
    'Contrast': ['contrast', 'concession', 'concession-e', 'antithesis', 'antithesis-e'],
    'Elaboration': ['elaboration-additional', 'elaboration-additional-e', 'elaboration-general-specific-e',
                    'elaboration-general-specific', 'elaboration-part-whole', 'elaboration-part-whole-e',
                    'elaboration-process-step', 'elaboration-process-step-e', 'elaboration-object-attribute-e',
                    'elaboration-object-attribute', 'elaboration-set-member', 'elaboration-set-member-e', 'example',
                    'example-e', 'definition', 'definition-e'],
    'Enablement': ['purpose', 'purpose-e', 'enablement', 'enablement-e'],
    'Evaluation': ['evaluation', 'evaluation-n', 'evaluation-s-e', 'evaluation-s', 'interpretation-n',
                   'interpretation-s-e', 'interpretation-s', 'interpretation', 'conclusion', 'comment', 'comment-e',
                   'comment-topic'],
    'Explanation': ['evidence', 'evidence-e', 'explanation-argumentative', 'explanation-argumentative-e', 'reason',
                    'reason-e'],
    'Joint': ['list', 'disjunction'],
    'Manner-Means': ['manner', 'manner-e', 'means', 'means-e'],
    'Topic-Comment': ['problem-solution', 'problem-solution-n', 'problem-solution-s', 'question-answer',
                      'question-answer-n', 'question-answer-s', 'statement-response', 'statement-response-n',
                      'statement-response-s', 'topic-comment', 'comment-topic', 'rhetorical-question'],
    'Summary': ['summary', 'summary-n', 'summary-s', 'restatement', 'restatement-e'],
    'Temporal': ['temporal-before', 'temporal-before-e', 'temporal-after', 'temporal-after-e', 'temporal-same-time',
                 'temporal-same-time-e', 'sequence', 'inverted-sequence'],
    'Topic-Change': ['topic-shift', 'topic-drift'],
    'Textual-Organization': ['textualorganization'],
    'span': ['span'],
    'Same-Unit': ['same-unit']
}

rel2class = {}
for cl, rels in class2rel.items():
    rel2class[cl.lower()] = cl
    for rel in rels:
        rel2class[rel.lower()] = cl
