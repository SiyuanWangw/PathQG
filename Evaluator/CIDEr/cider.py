from Appendix.Evaluator.CIDEr.cider_scorer import CiderScorer


class CIDErEvaluator:
    """
    Main Class to compute the CIDEr metric 

    """
    def __init__(self, test=None, refs=None, n=4, sigma=6.0):
        # set cider to sum over 1 to 4-grams
        self._n = n
        # set the standard deviation parameter for gaussian penalty
        self._sigma = sigma

    def compute_score(self, gts, res):
        """
        Main function to compute CIDEr score
        :param  hypo_for_image (dict) : dictionary with key <image> and value <tokenized hypothesis / candidate sentence>
                ref_for_image (dict)  : dictionary with key <image> and value <tokenized reference sentence>
        :return: cider (float) : computed CIDEr score for the corpus 
        """

        assert(gts.keys() == res.keys())
        imgIds = gts.keys()

        cider_scorer = CiderScorer(n=self._n, sigma=self._sigma)

        for id in imgIds:
            hypo = res[id]
            ref = gts[id]
            # Sanity check.
            assert(type(hypo) is list)
            # assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) > 0)
            for h in hypo:
                cider_scorer += (h, ref)

        (score, scores) = cider_scorer.compute_score()
        score = [score]
        return score

    def method(self):
        return "CIDEr"


def main():
    ref = {'1': ['I like it !', 'I love it !'], '2': ['I completely do not know !'],
           '3': ['how about you ?'], '4': ['what is this ?'], 5: ['this is amazing !']}
    hypo = {'1': ['I love you !'], '2': ['I do not know !'], '3': ['how are you ?'],
            '4': ['what is this animal ?'], 5: ['this is awkward !']}
    meteor = CIDErEvaluator()
    score = meteor.compute_score(ref, hypo)
    print(score, type(score))


if __name__ == '__main__':
    main()