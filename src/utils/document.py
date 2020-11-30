from os.path import isfile

from utils.token import Token

class Doc(object):
    """ Build one doc instance from *.merge file
    """

    def __init__(self):
        """
        """
        self.token_dict = None
        self.edu_dict = None
        self.fmerge = None

    def read_from_fmerge(self, fmerge):
        """ Read information from the merge file, and create an Doc instance
        :type fmerge: string
        :param fmerge: merge file name
        """
        self.fmerge = fmerge
        if not isfile(fmerge):
            raise IOError("File doesn't exist: {}".format(fmerge))
        gidx, self.token_dict = 0, {}
        with open(fmerge, 'r') as fin:
            for line in fin:
                line = line.strip()
                if len(line) == 0:
                    continue
                tok = self._parse_fmerge_line(line)
                self.token_dict[gidx] = tok
                gidx += 1
        fedus = fmerge.replace(".merge", ".edus")
        self.doc_edus = []
        with open(fedus, 'r') as fin:
            for line in fin:
                line = line.strip()
                if len(line) == 0:
                    continue
                self.doc_edus.append(line)
        # Get EDUs from tokendict
        self.edu_dict = self._recover_edus(self.token_dict)

    @staticmethod
    def _parse_fmerge_line(line):
        """ Parse one line from *.merge file
        """
        items = line.split("\t")
        tok = Token()
        tok.pidx, tok.sidx, tok.tidx = int(items[-1]), int(items[0]), int(items[1])
        # Without changing the case
        tok.word = items[2]
        try:
            tok.eduidx = int(items[9])
        except ValueError:
            print("EDU index for {} is missing in fmerge file".format(tok.word))
            pass
        return tok

    @staticmethod
    def _recover_edus(token_dict):
        """ Recover EDUs from token_dict
        """
        N, edu_dict = len(token_dict), {}
        for gidx in range(N):
            token = token_dict[gidx]
            eidx = token.eduidx
            try:
                val = edu_dict[eidx]
                edu_dict[eidx].append(gidx)
            except KeyError:
                edu_dict[eidx] = [gidx]
        return edu_dict


if __name__ == '__main__':
    doc = Doc()
    fmerge = "../data/data_dir/train_dir/wsj_0603.out.merge"
    doc.read_from_fmerge(fmerge)
    print(len(doc.edu_dict))
    print(doc.edu_dict)
