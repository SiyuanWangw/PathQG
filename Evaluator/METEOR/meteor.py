import os
import sys
import subprocess
import threading

# Assumes meteor-1.5.jar is in the same directory as meteor.py.  Change as needed.
METEOR_JAR = 'meteor-1.5.jar'
METEOR_JAR4 = 'meteor-1.4.jar'
# print METEOR_JAR


class Meteor:
    def __init__(self):
        self.lock = threading.Lock()
        self.meteor_cmd = ['java', '-jar', '-Xmx2G', METEOR_JAR, '-', '-', '-stdio', '-l', 'en', '-norm']
        self.meteor_p = subprocess.Popen(self.meteor_cmd, cwd=os.path.dirname(os.path.abspath(__file__)),
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                         bufsize=0, encoding='utf-8')
        # Used to guarantee thread safety

    def compute_score(self, gts, res):
        assert (gts.keys() == res.keys())
        imgIds = gts.keys()
        scores = []

        eval_line = 'EVAL'
        self.lock.acquire()
        for i in imgIds:
            for h in res[i]:
                stat = self._stat(h, gts[i])
                eval_line += ' ||| {}'.format(stat)

        self.meteor_p.stdin.write('{}\n'.format(eval_line))
        for i in imgIds:
            for _ in range(len(res[i])):
                scores.append(float(self.meteor_p.stdout.readline().strip()))
        score = self.meteor_p.stdout.readline().strip()
        self.lock.release()
        return [float(score)], scores

    def method(self):
        return "METEOR"

    def _stat(self, hypothesis_str, reference_list):
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str)) + '\n'
        # self.meteor_p.stdin.write('{}'.format(score_line.encode('utf-8')))
        self.meteor_p.stdin.write(score_line)
        # out = self.meteor_p.stdout
        # print(self.meteor_p.stdout)
        return self.meteor_p.stdout.readline().strip()

    def _score(self, hypothesis_str, reference_list):
        self.lock.acquire()
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        self.meteor_p.stdin.write('{}\n'.format(score_line))
        stats = self.meteor_p.stdout.readline().strip()
        eval_line = 'EVAL ||| {}'.format(stats)
        # EVAL ||| stats
        self.meteor_p.stdin.write('{}\n'.format(eval_line))
        score = float(self.meteor_p.stdout.readline().strip())
        # bug fix: there are two values returned by the jar file, one average, and one all, so do it twice
        # thanks for Andrej for pointing this out
        score = float(self.meteor_p.stdout.readline().strip())
        self.lock.release()
        return score

    def __del__(self):
        self.lock.acquire()
        self.meteor_p.stdin.close()
        self.meteor_p.kill()
        self.meteor_p.wait()
        self.lock.release()


def main():
    hypo = {'1': ['I like it !', 'I love you !', 'I know it'], '2': ['I completely do not know !'],
            '3': ['how about you ?'], '4': ['what is this ?'], 5: ['this is amazing !']}
    ref = {'1': ['I love you !'], '2': ['I do not know !'], '3': ['how are you ?'],
           '4': ['what is this animal ?'], 5: ['this is awkward !']}
    meteor = Meteor()
    score = meteor.compute_score(ref, hypo)
    print(score)
    pass


if __name__ == '__main__':
    main()