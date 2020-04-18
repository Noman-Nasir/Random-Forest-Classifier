
import tree as tree
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.model_selection import train_test_split


class RandomForest:
    ''' Implements the Random Forest For Classification... '''

    def __init__(self, ntrees=10, treedepth=5, usebagging=False, baggingfraction=0.6,
                 weaklearner="Conic",
                 nsplits=10,
                 nfeattest=None, posteriorprob=False, scalefeat=True):
        """
            Build a random forest classification forest....

            Input:
            ---------------
                ntrees: number of trees in random forest
                treedepth: depth of each tree
                usebagging: to use bagging for training multiple trees
                baggingfraction: what fraction of training set to use for building each tree,
                weaklearner: which weaklearner to use at each interal node, e.g. "Conic, Linear, Axis-Aligned, Axis-Aligned-Random",
                nsplits: number of splits to test during each feature selection round for finding best IG,
                nfeattest: number of features to test for random Axis-Aligned weaklearner
                posteriorprob: return the posteriorprob class prob
                scalefeat: wheter to scale features or not...
        """

        self.ntrees = ntrees
        self.treedepth = treedepth
        self.usebagging = usebagging
        self.baggingfraction = baggingfraction

        self.weaklearner = weaklearner
        self.nsplits = nsplits
        self.nfeattest = nfeattest

        self.posteriorprob = posteriorprob

        self.scalefeat = scalefeat

        pass

    def findScalingParameters(self, X):
        """
            find the scaling parameters
            input:
            -----------------
                X= m x d training data matrix...
        """
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def applyScaling(self, X):
        """
            Apply the scaling on the given training parameters
            Input:
            -----------------
                X: m x d training data matrix...
            Returns:
            -----------------
                X: scaled version of X
        """
        X = X - self.mean
        X = X / self.std
        return X

    def train(self, X, Y, vX=None, vY=None):
        '''
        Trains a RandomForest using the provided training set..

        Input:
        ---------
        X: a m x d matrix of training data...
        Y: labels (m x 1) label matrix

        vX: a n x d matrix of validation data (will be used to stop growing the RF)...
        vY: labels (n x 1) label matrix

        Returns:
        -----------

        '''

        nexamples, nfeatures = X.shape

        self.findScalingParameters(X)
        if self.scalefeat:
            X = self.applyScaling(X)

        self.trees = []


        if vX is not None and vY is not None:
          self.ntrees = self.find_best_parameters(X, Y, vX, vY)

        print("\nBuilding Classifier\n")
        for ntree in range(self.ntrees):
            print ("Creating tree # {}".format(ntree+1))
            self.trees.append(self.train_tree(X, Y))
            print('')

    def train_tree(self, X, Y, verbose = True ):
        '''
        Trains A tree based on given arguments

        return : the Decision Tree object
        '''

        dt = tree.DecisionTree(exthreshold=10, maxdepth=self.treedepth,
                               weaklearner=self.weaklearner, nsplits=self.nsplits)
        dt.verbose = verbose

        if self.usebagging:
          X_train, _, Y_train, _ = train_test_split(X, Y, train_size=self.baggingfraction)
          dt.train(X_train, Y_train)
          return dt
        
        dt.train(X, Y)
        return dt
    
    def find_best_parameters(self, X, Y, vX, vY):
      '''

        Trains RandomForest using the provided training set with trees ranging from
        10 to 30 return the best ntrees with best accuracy.

        Input:
        ---------
        X: a m x d matrix of training data...
        Y: labels (m x 1) label matrix

        vX: a n x d matrix of validation data (will be used to stop growing the RF)...
        vY: labels (n x 1) label matrix

        Returns:
        -----------
        Optimal Number of Trees
        Optimal Depth

      '''

      accuracy_list = []
      optimal_parameters = []

      for ntrees in range(10,31,2):

        print ("Creating Classifier with {} trees.".format(ntrees))
        # Training Classifier
        for ntree in range(ntrees):
          self.trees.append(self.train_tree(X, Y, verbose = False))

        Yp = self.predict(vX)
        optimal_parameters.append( ntrees )
        acc =  self.find_accuracy(vY,Yp)
        print ("Accuracy of Classifier with {} trees is {}.".format(ntrees, acc*100))
        accuracy_list.append(acc)
        self.trees.clear()

      maxAcc = np.argmax(accuracy_list)
      param = optimal_parameters[maxAcc]
      print("Validation Comple. Optimal Parameters: Trees =  {} with Accuracy {} ".format(param,accuracy_list[maxAcc]) )
      return param

    def find_accuracy(self, Y, Yp):
      plabels = pd.Series(np.squeeze(Yp))
      tlabels = pd.Series(np.squeeze(Y))

      acc = np.sum(tlabels == plabels) / len(Y)
      return acc

    def predict(self, X):
        """
        Test the trained RF on the given set of examples X


            Input:
            ------
                X: [m x d] a d-dimensional test examples.

            Returns:
            -----------
                pclass: the predicted class for the given example, i.e. to which it belongs
        """
        z = []

        if self.scalefeat:
            X = self.applyScaling(X)

        pred = []

        for tree in self.trees:
            z.append(tree.predict(X))
        z = np.array(z).T

        for row in z:
            pred.append(stats.mode(row)[0])
        return pred
