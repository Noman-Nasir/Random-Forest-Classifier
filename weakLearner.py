import tools as t  # set of tools for plotting, data splitting, etc..
from collections import defaultdict  # default dictionary
import time
import pandas as pd
import numpy as np
import scipy.stats as stats


class WeakLearner:  # A simple weaklearner you used in Decision Trees...

    """ 
    A Super class to implement different forms of weak learners...
    """

    def __init__(self):
        
        pass

    def train(self, X, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible split points for
            possible feature selection

            Input:
            ---------
            feat: a contiuous feature
            Y: labels

            Returns:
            ----------
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node

        '''
        nexamples, nfeatures = X.shape

        gain = Xlidx = Xridx = None

        min_gain = +np.inf

        for feat in range(nfeatures):
            tup = self.evaluate_numerical_attribute(X[:, feat], Y)

            if tup[1] < min_gain:
                splitValue, gain, Xlidx, Xridx = tup
                min_gain = gain
                self.split = splitValue
                self.feat_indx = feat

        return gain, Xlidx, Xridx

    def __entropy(self, Y):
        _, counts = np.unique(Y, return_counts=True)
        probs = np.array(counts/len(Y))

        entropy = -np.sum(probs * np.log2(probs))

        return entropy

    def evaluate(self, X):
        """
        Evalute the trained weak learner  on the given example...
        """
        
        if X[self.feat_indx] < self.split:
            return True
        return False

    def evaluate_numerical_attribute(self, feat, Y):
        '''
            Evaluates the numerical attribute for all possible split points for
            possible feature selection

            Input:
            ---------
            feat: a contiuous feature
            Y: labels

            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node

        '''

        classes = np.unique(Y)
        
        sorted_feat = feat.copy()
        sorted_feat.sort()

        # splits = (sorted_feat[1:] + sorted_feat[:-1]) / 2

        q1 = np.quantile(feat, 0.25)
        q3 = np.quantile(feat, 0.75)

        # print(q1, q3)

        mini = np.argpartition(feat, 2)[:2]
        dis = feat[mini[1]] - feat[mini[0]]

        # print(dis)

        splits = np.arange(q1, q3, dis)

        # print("Splits size = {}".format(splits.shape))

        # print("Uniq Splits size = {}".format(np.unique(splits).shape))

        gain = []
        for split in splits:

            left = Y[feat < split]
            right = Y[feat >= split]

            ent_left = self.__entropy(left)
            ent_right = self.__entropy(right)
            wet_ent = (len(left)/len(Y))*ent_left + \
                (len(right)/len(Y))*ent_right

            gain.append(wet_ent)

        ind = np.argmin(gain)
        split = splits[ind]
        mingain = gain[ind]

        Xlidx = feat < split
        Xridx = np.invert(Xlidx)

        return split, mingain, Xlidx, Xridx


class RandomWeakLearner(WeakLearner):  # Axis Aligned weak learner....
    """ 
    An Inherited class to implement Axis-Aligned weak learner using 
        a random set of features from the given set of features...

    """

    def __init__(self, nsplits=+np.inf, nrandfeat=None):
        """
        Input:
            nsplits = How many nsplits to use for each random feature, (if +np.inf, check all possible splits)
            nrandfeat = number of random features to test for each node (if None, nrandfeat= sqrt(nfeatures) )
        """
        WeakLearner.__init__(self)  # calling base class constructor...
        self.nsplits = nsplits
        self.nrandfeat = nrandfeat
        pass

    def train(self, X, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible split points for
            possible feature selection

            Input:
            ---------
            X: a [m x d]  features matrix
            Y: a [m x 1] labels matrix

            Returns:
            ----------
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node

        '''
        nexamples, nfeatures = X.shape

        if(not self.nrandfeat):
            self.nrandfeat = int(np.round(np.sqrt(nfeatures)))


        randfeat = np.random.randint(nfeatures, size=self.nrandfeat)

        minscore = bXl = bXr = None
        max_gain = +np.inf

        for feat in randfeat:
            tup = self.findBestRandomSplit(X[:, feat], Y)

            if tup[1] < max_gain:
                splitValue, minscore, bXl, bXr = tup
                max_gain = minscore
                self.split = splitValue
                self.feat_indx = feat

        return minscore, bXl, bXr

    def findBestRandomSplit(self, feat, Y):
        """

            Find the best random split by randomly sampling "nsplits"
            splits from the feature range...

            Input:
            ----------
            feat: [n X 1] nexamples with a single feature
            Y: [n X 1] label vector...

        """
        frange = np.max(feat)-np.min(feat)


        if self.nsplits == +np.inf:
            splitvalue, minscore, Xlidx, Xridx = WeakLearner.evaluate_numerical_attribute(
                self, feat, Y)
        else:
            splits = np.random.randint(
                np.floor(np.min(feat)) - 1, np.ceil(np.max(feat)) + 1, self.nsplits)
            entropy = []
            for split in splits:
                entropy.append(self.calculateEntropy(Y, feat < split))

            ind = np.argmin(entropy)
            splitvalue = splits[ind]
            minscore = entropy[ind]
            Xlidx = feat < splitvalue
            Xridx = np.invert(Xlidx)

        return splitvalue, minscore, Xlidx, Xridx

    def calculateEntropy(self, Y, mship):
        """
            calculates the split entropy using Y and mship (logical array) telling which 
            child the examples are being split into...

            Input:
            ---------
                Y: a label array
                mship: (logical array) telling which child the examples are being split into, whether
                        each example is assigned to left split or the right one..
            Returns:
            ---------
                entropy: split entropy of the split
        """

        lexam = Y[mship]
        rexam = Y[np.logical_not(mship)]

        pleft = len(lexam) / float(len(Y))
        pright = 1-pleft

        pl = np.array(np.unique(lexam, return_counts=True)) / \
            float(len(lexam)) + np.spacing(1)
        pr = np.array(np.unique(rexam, return_counts=True)) / \
            float(len(rexam)) + np.spacing(1)

        hl = -np.sum(pl*np.log2(pl))
        hr = -np.sum(pr*np.log2(pr))

        sentropy = pleft * hl + pright * hr

        return sentropy


# build a classifier ax+by+c=0
# A 2-dimensional linear weak learner....
class LinearWeakLearner(RandomWeakLearner):
    """ An Inherited class to implement 2D line based weak learner using 
        a random set of features from the given set of features...


    """

    def __init__(self, nsplits=10):
        """
        Input:
            nsplits = How many splits to use for each choosen line set of parameters...

        """
        RandomWeakLearner.__init__(self, nsplits)

        pass

    def train(self, X, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible 

            Input:
            ---------
            X: a [m x d] data matrix ...
            Y: labels

            Returns:
            ----------
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node

        '''
        nexamples, nfeatures = X.shape


        if(not self.nrandfeat):
            self.nrandfeat = int(np.round(np.sqrt(nfeatures)))

        randfeat_pairs = self.__find_random_feature_pairs(nfeatures)

        minscore = bXl = bXr = None
        min_gain = +np.inf

        for pair in randfeat_pairs:
            tup = self.find_best_linear_split(X[:, pair], Y)

            if tup[1] < min_gain:
                splitValue, minscore, bXl, bXr = tup
                min_gain = minscore
                self.feat_pair = pair
                self.split = splitValue


        return minscore, bXl, bXr

    def evaluate(self, X):
        """
        Evalute the trained weak learner  on the given example...
        """

        vec = X[self.feat_pair]
        # ax+by+c
        dp = np.dot(vec, self.split[:2]) + self.split[2]
        if dp < 0:
            return True
        return False

    def find_best_linear_split(self, feat, Y):

        lower = np.floor(feat.min())
        upper = np.ceil(feat.max())
        def rand_theta(): return np.random.randint(lower, upper, 3)
        splits = [rand_theta() for _ in range(self.nsplits)]

        entropy = []
        dot_products = []
        for split in splits:

            dp = np.dot(feat, split[:2]) + split[2]
            lchild = dp < 0
            entropy.append(RandomWeakLearner.calculateEntropy(self, Y, lchild))

        ind = np.argmin(entropy)
        splitvalue = splits[ind]
        minscore = entropy[ind]
        dp = np.dot(feat, splitvalue[:2]) + splitvalue[2]
        Xlidx = dp < 0
        Xridx = np.invert(Xlidx)

        return splitvalue, minscore, Xlidx, Xridx

    def __find_random_feature_pairs(self, nfeatures):
        randfeat = np.arange(nfeatures)
        np.random.shuffle(randfeat)
        feat_pairs = np.array(list(zip(randfeat[:-1], randfeat[1:])))
        rndm_indx = np.random.choice(feat_pairs.shape[0], self.nrandfeat)
        randfeat_pairs = feat_pairs[rndm_indx]
        return randfeat_pairs


# build a classifier ax2+bxy+cy2+dx+ey+f=0
class ConicWeakLearner(LinearWeakLearner):
    def __init__(self, nsplits=10):
        LinearWeakLearner.__init__(self, nsplits)

    def train(self, X, Y):

        nexamples, nfeatures = X.shape

        if(not self.nrandfeat):
            self.nrandfeat = int(np.round(np.sqrt(nfeatures)))

        randfeat_pairs = self.__find_random_feature_pairs(nfeatures)

        minscore = bXl = bXr = None
        min_gain = +np.inf

        for pair in randfeat_pairs:
            tup = self.find_best_conic_split(X[:, pair], Y)

            if tup[1] < min_gain:
                splitValue, minscore, bXl, bXr = tup
                min_gain = minscore
                self.split = splitValue
                self.feat_pair = pair

        return minscore, bXl, bXr

    def find_best_conic_split(self, feat, Y):

        lower = np.floor(feat.min())
        upper = np.ceil(feat.max())
        def rand_theta(): return np.random.randint(lower, upper, 6)
        splits = [rand_theta() for _ in range(self.nsplits)]

        x1 = feat[:, 0]
        x2 = feat[:, 1]
        x_vec = np.vstack((x1**2, x1*x2, x2**2, x1, x2, np.ones(x1.shape))).T

        entropy = []
        dot_products = []
        for split in splits:

            dp = np.dot(x_vec, split)

            lchild = dp < 0
            entropy.append(LinearWeakLearner.calculateEntropy(self, Y, lchild))

        ind = np.argmin(entropy)
        splitvalue = splits[ind]
        minscore = entropy[ind]
        dp = np.dot(x_vec, splitvalue)
        Xlidx = dp < 0
        Xridx = np.invert(Xlidx)

        return splitvalue, minscore, Xlidx, Xridx

    def __find_random_feature_pairs(self, nfeatures):
        randfeat = np.arange(nfeatures)
        np.random.shuffle(randfeat)
        feat_pairs = np.array(list(zip(randfeat[:-1], randfeat[1:])))
        rndm_indx = np.random.choice(feat_pairs.shape[0], self.nrandfeat)
        randfeat_pairs = feat_pairs[rndm_indx]
        return randfeat_pairs

    def evaluate(self, X):
        x1, x2 = X[self.feat_pair]
        x_vec = np.vstack((x1**2, x1*x2, x2**2, x1, x2, 1)).T
#       Conic dot product
        dp = np.dot(x_vec, self.split)
        if dp < 0:
            return True
        return False
