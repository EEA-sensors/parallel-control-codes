"""
Text correction model for Viterbi testing.

@author: Simo Särkkä
"""

import numpy as np

class TextModel:
    def __init__(self, chars=' abcdefghijklmnopqrstuvwxyz', errp=0.1, order=1):
        """ Constructor.

        Parameters:
            chars: String of characters.
            errp: Error probability.
            order: Markov order.
        """
        self.chars = chars
        self.errp = errp

        char_dict = {}
        for i in range(len(chars)):
            char_dict[chars[i]] = i
        self.char_dict = char_dict

        self.order = order

        cond_chars = []
        cond_dict = {}
        index_list = order * [0]
        for i in range(len(chars) ** order):
            str = ''
            for j in range(len(index_list)):
                str = str + self.chars[index_list[j]]
            cond_dict[str] = i
            cond_chars.append(str)

            done = False
            j = len(index_list) - 1
            while not done and j >= 0:
                index_list[j] = index_list[j] + 1
                if index_list[j] >= len(self.chars):
                    index_list[j] = 0
                    j = j - 1
                else:
                    done = True
        self.cond_dict = cond_dict
        self.cond_chars = cond_chars


    def get1stOrderPo(self):
        """ Get 1st order output matrix.

        Returns:
            Po: Output matrix.
        """
        Po = np.zeros((len(self.chars), len(self.chars)), dtype=float)
        pr = np.ones((len(self.chars),), dtype=float)
        pr = pr / pr.sum()
        for i in range(len(self.chars)):
            Po[i, i] = (1.0 - self.errp)
            Po[i, :] = Po[i, :] + self.errp * pr
        return Po

    def get1stOrderPi(self, train_text):
        """ Get 1st order transition matrix.

        Parameters:
            train_text: Training text.

        Returns:
            Pi: Transition matrix.
        """

        Pi = np.ones((len(self.chars), len(self.chars)), dtype=float)

        c1 = train_text[0]
        for i in range(1, len(train_text)):
            c0 = c1
            c1 = train_text[i]
            i0 = self.char_dict[c0]
            i1 = self.char_dict[c1]
            Pi[i0, i1] = Pi[i0, i1] + 1

        for i in range(Pi.shape[0]):
            Pi[i, :] = Pi[i, :] / Pi[i, :].sum()

        return Pi

    def get1stOrderPrior(self, train_text):
        """ Get 1st order prior.

        Parameters:
            train_text: Training text.

        Returns:
            prior: Prior.
        """
        prior = np.ones((len(self.chars),), dtype=float)
        for i in range(len(train_text)):
            c = train_text[i]
            prior[self.char_dict[c]] = prior[self.char_dict[c]] + 1.0
        prior = prior / prior.sum()

        return prior


    def getNthOrderPo(self):
        """ Get Nth order output matrix.

        Returns:
            Po: Output matrix.
        """
        Po = self.get1stOrderPo()
        if self.order > 1:
            Po = np.tile(Po, (len(self.chars) ** (self.order-1), 1))

        return Po


    def getNthOrderPi(self, train_text):
        """ Get Nth order transition matrix.

        Parameters:
            train_text: Training text.

        Returns:
            Pi: Transition matrix.
        """
        Pi = 1e-12 * np.ones((len(self.chars)**self.order, len(self.chars)**self.order), dtype=float)

        for i in range(Pi.shape[0]):
            for j in range(Pi.shape[0]):
                c0 = self.cond_chars[i]
                c1 = self.cond_chars[j]
                if c0[1:] == c1[:-1]:
                    Pi[i,j] = Pi[i,j] + 1

        c1 = train_text[:self.order]
        for i in range(self.order, len(train_text)):
            c0 = c1
            c1 = c1[1:] + train_text[i]
            i0 = self.cond_dict[c0]
            i1 = self.cond_dict[c1]
            Pi[i0, i1] = Pi[i0, i1] + 1

        for i in range(Pi.shape[0]):
            Pi[i, :] = Pi[i, :] / Pi[i, :].sum()

        return Pi


    def getNthOrderPrior(self, train_text):
        """ Get Nth order prior.

        Parameters:
            train_text: Training text.

        Returns:
            prior: Prior.
        """
        prior = np.ones((len(self.chars)**self.order,), dtype=float)
        for i in range(len(train_text)-self.order+1):
            c = train_text[i:(i + self.order)]
            prior[self.cond_dict[c]] = prior[self.cond_dict[c]] + 1.0
        prior = prior / prior.sum()

        return prior


    def makeNoisyText(self, test_text):
        """ Make noisy text for testing.

        Parameters:
            test_text: Test text.

        Returns:
            text: Noisy text.
        """

        Po = self.get1stOrderPo()

        text = ''
        for i in range(len(test_text)):
            c = test_text[i]
            j = self.char_dict[c]
            p = Po[j, :]
            text = text + self.chars[np.random.choice(a=p.shape[0], size=1, p=p)[0]]

        return text

    def textToIndex(self, text):
        """ Convert text to character indices.

        Parameters:
            text: Text.

        Returns:
            ys: Character indices.
        """
        ys = np.zeros((len(text),), dtype=int)

        for i in range(len(text)):
            c = text[i]
            ys[i] = self.char_dict[c]

        return ys

    def indexToText(self, ys):
        """ Convert character indices to text.

        Parameters:
            ys: Character indices.

        Returns:
            text: Text.
        """
        text = ''
        for i in range(ys.shape[0]):
            j = ys[i] % len(self.chars)
            text = text + self.chars[j]

        return text


