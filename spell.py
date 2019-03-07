'''Spell checker to find the closest word.'''

from collections import Counter
import string

class SpellChecker(object):
    def __init__(self, wordlist, max_distance=1):
        self.words = Counter(wordlist)
        self.n = len(wordlist)
        self.max_distance = max_distance

        self.p_substitution = 4./10000
        self.p_deletion = 1./10000
        self.p_insertion = 2./10000

    def p(self, w):
        return float(self.words[w]) / self.n

    def __call__(self, w):
        return self.correct(w)

    def correct(self, w, max_distance=None):
        if max_distance is None:
            max_distance = self.max_distance
        return self.max_score(w, max_distance)[0]

    def max_score(self, w, max_distance):
        if max_distance == 0:
            return w, self.p(w)

        score = self.p(w)
        word = w

        # substitutions
        for i in range(len(w)):
            for c in string.ascii_lowercase:
                q = w[:i] + c + w[i+1:]
                qword, qscore = self.max_score(q, max_distance=max_distance-1)
                qscore *= self.p_substitution
                if qscore > score:
                    score = qscore
                    word = qword

        # deletions
        for i in range(len(w)):
            q = w[:i] + w[i+1:]
            qword, qscore = self.max_score(q, max_distance=max_distance-1)
            qscore *= self.p_deletion
            if qscore > score:
                score = qscore
                word = qword

        # insertions
        for i in range(0, len(w)+1):
            for c in string.ascii_lowercase:
                q = w[:i] + c + w[i:]
                qword, qscore = self.max_score(q, max_distance=max_distance-1)
                qscore *= self.p_insertion
                if qscore > score:
                    score = qscore
                    word = qword

        return word, score
