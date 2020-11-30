from utils.document import Doc
from models.tree import RstTree
from eval.metrics import Metrics
from models.state import ParsingState
from math import isclose

def test_doc():
    doc = Doc()
    fmerge = "../data/data_dir/train_dir/wsj_0603.out.merge"
    doc.read_from_fmerge(fmerge)
    # Correct number of tokens/edus
    assert len(doc.edu_dict) == 7
    edu_lens = [10, 3, 9, 7, 9, 13, 7]
    for i in range(1, 8):
        new_edu = []
        for tok_idx in doc.edu_dict[i]:
            new_word = doc.token_dict[tok_idx].word
            new_edu.append(new_word)
        assert len(doc.edu_dict[i]) == edu_lens[i-1]
    
    num_tokens = sum(edu_lens)
    
    # Sidx/pidx/ TODO eduidx
    assert len(doc.token_dict) == num_tokens
    assert doc.token_dict[0].sidx == 0 and doc.token_dict[num_tokens - 1].sidx == 2
    assert doc.token_dict[37].sidx == 1 and doc.token_dict[38].sidx == 2
    assert doc.token_dict[0].pidx == 1 and doc.token_dict[num_tokens - 1].pidx == 2
    assert doc.token_dict[0].eduidx == 1 and doc.token_dict[num_tokens - 1].eduidx == 7
    # Bigger doc
    doc = Doc()
    fmerge = "../data/data_dir/train_dir/wsj_0606.out.merge"
    doc.read_from_fmerge(fmerge)
    assert len(doc.edu_dict) == 57
    assert doc.token_dict[127].pidx == 2 and doc.token_dict[128].pidx == 3
    assert doc.token_dict[0].eduidx == 1 and doc.token_dict[581].eduidx == 57
    
    assert doc.token_dict[362].eduidx == 38 and doc.token_dict[363].eduidx == 39

def test_tree():
    fdis = "../data/data_dir/train_dir/wsj_0603.out.dis"
    fmerge = "../data/data_dir/train_dir/wsj_0603.out.merge"
    tree = RstTree(fdis, fmerge)
    tree.build()
    bftbin = RstTree.BFTbin(tree.tree)
    assert len(bftbin) == 13
    edu_spans = [(1, 7), (1, 5), (6, 7), (1, 4), (5, 5), (6, 6), (7, 7), (1, 2), (3, 4), (1, 1), (2, 2), (3, 3), (4, 4)]
    r, n, s = 'Root', 'Nucleus', 'Satellite'
    props = [r, n, n, n, s, n, s, n, n, n, s, n, s] 
    rels = [None, 'TextualOrganization', 'TextualOrganization', 
            'span', 'elaboration-additional', 'span', 'attribution', 'Same-Unit', 
            'Same-Unit', 'span', 'elaboration-object-attribute-e', 'span', 
            'elaboration-object-attribute-e',]
    for i, span in enumerate(bftbin):
        assert span.edu_span == edu_spans[i]
        assert span.prop == props[i]
        assert span.relation == rels[i]
    
    # TODO: Check where relation conversion happens? In RST-Coref paper, in generate_action_samples
    fdis, action_seq = tree.generate_action_samples()
    assert fdis == fdis
    sh, rd = 'Shift', 'Reduce' 
    nn, ns, sn = 'NN', 'NS', 'SN'
    gold_action_seq = [(sh, None), (sh, None), (rd, ns), (sh, None), (sh, None),
                  (rd, ns), (rd, nn), (sh, None), (rd, ns), (sh, None),
                  (sh, None), (rd, ns), (rd, nn)]
    assert len(action_seq) == len(gold_action_seq)
    for i in range(len(action_seq)):
        assert action_seq[i] == gold_action_seq[i]
        
    # Check that tree binarization is node correctly
    fdis = "../data/data_dir/train_dir/file2.dis"
    fmerge = "../data/data_dir/train_dir/file2.merge"
    tree = RstTree(fdis, fmerge)
    tree.build()
    spans = [span.edu_span for span in RstTree.BFTbin(tree.tree)]
    nonbinary_spans = [(22, 28), (22, 22), (23, 28), (23, 23), (24, 28), 
                       (24, 24), (25, 28), (25, 25), (26, 28), (26, 26), 
                       (27, 28), (27, 27), (28, 28), (11, 15), (11, 11), 
                       (12, 15), (12, 12), (13, 15), (13, 14), (13, 13),
                       (14, 14), (15, 15), (45, 57), (45, 47), (48, 57),
                       (48, 53), (54, 57), (54, 56), (57, 57)]
    for span in nonbinary_spans:
        assert span in spans

def eval_trees(silver_tree, gold_tree, rst_span_perf, rst_nuc_perf, span_perf, nuc_perf):
    # RST-Parseval
    met = Metrics(use_parseval=False, levels=['span', 'nuclearity'])
    met.eval(silver_tree, gold_tree)
    assert met.span_perf.hit_num / met.span_num == rst_span_perf
    assert met.nuc_perf.hit_num / met.span_num == rst_nuc_perf
    
    # Parseval
    met = Metrics(use_parseval=True, levels=['span', 'nuclearity'])
    met.eval(silver_tree, gold_tree)
    assert met.span_perf.hit_num / met.span_num == span_perf
    assert met.nuc_perf.hit_num / met.span_num == nuc_perf
    
def test_eval():
    fdis = "../data/data_dir/train_dir/wsj_0603.out.dis"
    fmerge = "../data/data_dir/train_dir/wsj_0603.out.merge"
    gold_tree = RstTree(fdis, fmerge)
    gold_tree.build()
    eval_trees(gold_tree, gold_tree, 1, 1, 1, 1)  
    
    sr_parser = ParsingState([], [])
    sr_parser.init(gold_tree.doc)
    sh, rd = 'Shift', 'Reduce' 
    nn, ns, sn = 'NN', 'NS', 'SN'
    
    # Last two nuclearity actions are incorrect
    silver_nuc = [(sh, None), (sh, None), (rd, ns), (sh, None), (sh, None),
                  (rd, ns), (rd, nn), (sh, None), (rd, ns), (sh, None),
                  (sh, None), (rd, sn), (rd, ns)]
    
    for idx, action in enumerate(silver_nuc):
        sr_parser.operate(action)
    tree = sr_parser.get_parse_tree()
    silver_tree = RstTree()
    silver_tree.assign_tree(tree)
    silver_tree.assign_doc(gold_tree.doc)
    silver_tree.back_prop(tree, gold_tree.doc)
    
    eval_trees(silver_tree, gold_tree, 1, 9/12, 1, 4/6)  

    # Last two nuclearity actions are incorrect
    sr_parser = ParsingState([], [])
    sr_parser.init(gold_tree.doc)
    silver_nuc = [(sh, None), (sh, None), (sh, None), (rd, ns), (sh, None),
                  (rd, ns), (rd, nn), (sh, None), (rd, ns), (sh, None), (sh, None),
                  (rd, ns), (rd, ns)]
    
    for idx, action in enumerate(silver_nuc):
        sr_parser.operate(action)
    tree = sr_parser.get_parse_tree()
    silver_tree = RstTree()
    silver_tree.assign_tree(tree)
    silver_tree.assign_doc(gold_tree.doc)
    silver_tree.back_prop(tree, gold_tree.doc)
    eval_trees(silver_tree, gold_tree, 10/12, 7/12, 4/6, 3/6)  
    
test_doc()
test_tree()
test_eval()
print("Tests passed")