import sys
sys.path.append('../..')
from Evaluator.BLEU.bleu_scorer import BleuScorer
from tqdm import tqdm


class Bleu:
    def __init__(self, n=4):
        # default compute Blue score up to 4
        self._n = n
        self._hypo_for_image = {}
        self.ref_for_image = {}

    def compute_score(self, gts, res):

        assert(gts.keys() == res.keys())
        imgIds = gts.keys()

        bleu_scorer = BleuScorer(n=self._n)
        for id in tqdm(imgIds):
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) >= 1)

            # for each_hypo in hypo:
            #     bleu_scorer += (each_hypo, ref)
            bleu_scorer += (hypo[0], ref)

        #score, scores = bleu_scorer.compute_score(option='shortest')
        score, scores = bleu_scorer.compute_score(option='closest', verbose=0)
        #score, scores = bleu_scorer.compute_score(option='average', verbose=1)

        # return (bleu, bleu_info)
        return score, scores


    # def compute_score_train(self, gts, res):
    #
    #     imgIds = res.keys()
    #     bleu_scorer = BleuScorer(n=self._n)
    #
    #     ref = gts
    #     for id in tqdm(imgIds):
    #         # for each_hypo in hypo:
    #         #     bleu_scorer += (each_hypo, ref)
    #         bleu_scorer += (res[id][0], ref)
    #
    #     #score, scores = bleu_scorer.compute_score(option='shortest')
    #     score, scores = bleu_scorer.compute_score(option='closest', verbose=0)
    #     #score, scores = bleu_scorer.compute_score(option='average', verbose=1)
    #
    #     # return (bleu, bleu_info)
    #     return score, scores

    def method(self):
        return "Bleu"