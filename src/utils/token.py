class Token(object):
    """ Token class
    """

    def __init__(self):
        # Paragraph index, Sentence index, token index (within sent)
        self.pidx, self.sidx, self.tidx = None, None, None
        # Word, Lemma
        self.word = None