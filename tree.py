
# A good heuristic is to choose sqrt(nfeatures) to consider for each node...
import weakLearner as wl
import numpy as np
import scipy.stats as stats

class Node:
    def __init__(self, klasslabel='', pdistribution=[], score=0, wlearner=None):
        """
               Input:
               --------------------------
               klasslabel: to use for leaf node
               pdistribution: posteriorprob class probability at the node
               score: split score
               weaklearner: which weaklearner to use this node, an object of WeakLearner class or its childs...

        """
        self.lchild = None
        self.rchild = None
        self.klasslabel = klasslabel
        self.pdistribution = pdistribution
        self.score = score
        self.wlearner = wlearner
        self.fidx = None

    def set_childs(self, lchild, rchild):
        """
        function used to set the childs of the node
        input:
            lchild: assign it to node left child
            rchild: assign it to node right child
        """

        self.lchild = lchild
        self.rchild = rchild

    def isleaf(self):
        """
            return true, if current node is leaf node
        """
        if self.lchild or self.rchild:
            return False
        return True

    def isless_than_eq(self, X):
        """
            This function is used to decide which child node current example
            should be directed to. i.e. returns true, if the current example should be
            sent to left child otherwise returns false.
        """

        return self.wlearner.evaluate(X)

    def get_str(self):
        """
            returns a string representing the node information...
        """
        if self.isleaf():
            return 'C(posterior={},class={},Purity={})'.format(self.pdistribution, self.klasslabel, self.purity)
        else:
            return 'I(Fidx={},Score={},Split={})'.format(self.fidx, self.score, self.split)


class DecisionTree:
    ''' Implements the Decision Tree For Classification With Information Gain
        as Splitting Criterion....
    '''

    def __init__(self, exthreshold=5, maxdepth=10,
                 weaklearner="Conic", pdist=False, nsplits=10, nfeattest=None):
        '''
        Input:
        -----------------
            exthreshold: Number of examples to stop splitting, i.e. stop if number examples at a given node are less than exthreshold
            maxdepth: maximum depth of tree upto which we should grow the tree. Remember a tree with depth=10
            has 2^10=1K child nodes.
            weaklearner: weaklearner to use at each internal node.
            pdist: return posterior class distribution or not...
            nsplits: number of splits to use for weaklearner
        '''
        self.maxdepth = maxdepth
        self.exthreshold = exthreshold
        self.weaklearner = weaklearner
        self.nsplits = nsplits
        self.pdist = pdist
        self.nfeattest = nfeattest
        # Whether or not to print during tree building
        self.verbose = True
        assert (weaklearner in ["Conic", "Linear",
                                "Axis-Aligned", "Axis-Aligned-Random"])

    def getWeakLearner(self):
        if self.weaklearner == "Conic":
            return wl.ConicWeakLearner(self.nsplits)
        elif self.weaklearner == "Linear":
            return wl.LinearWeakLearner(self.nsplits)
        elif self.weaklearner == "Axis-Aligned":
            return wl.WeakLearner()
        else:
            return wl.RandomWeakLearner(self.nsplits, self.nfeattest)

    def train(self, X, Y):
        ''' Train Decision Tree using the given
            X [m x d] data matrix and Y labels matrix

            Input:
            ------
            X: [m x d] a data matrix of m d-dimensional examples.
            Y: [m x 1] a label vector.

            Returns:
            -----------
            Nothing
            '''
        nexamples, nfeatures = X.shape

        self.pdist = self.__get_pdist(Y)
        self.tree = self.build_tree(X, Y, self.maxdepth)


    def __entropy(self, Y):
        _, counts = np.unique(Y, return_counts=True)
        probs = np.array(counts/len(Y))

        entropy = -np.sum(probs * np.log2(probs))

        return entropy

    def build_tree(self, X, Y, depth):
        """

            Function is used to recursively build the decision Tree

            Input
            -----
            X: [m x d] a data matrix of m d-dimensional examples.
            Y: [m x 1] a label vector.

            Returns
            -------
            root node of the built tree...


        """
        nexamples, nfeatures = X.shape


        if nexamples == 0:
            return None

        if self.__entropy(Y) == 0:

            pdist = self.__get_pdist(Y)
            if self.verbose:
                print('Creating Child Node With ClassLabel={}, nexamples={}, classes={}'
                      .format(pdist[0][0], len(Y), pdist))

            leaf = Node(klasslabel=pdist[0][0], pdistribution=pdist)
            leaf.purity = 1.0
            return leaf

        if nexamples < self.exthreshold or depth == 1:
            pdist = self.__get_pdist(Y)

            purity = pdist[0][1]
            label = pdist[0][0]
            if self.verbose:
                print('Creating Child Node With ClassLabel={}, nexamples={}, classes={}'
                      .format(label, len(Y), pdist))

            leaf = Node(klasslabel=label, pdistribution=pdist)
            leaf.purity = purity
            return leaf

        wlrner = self.getWeakLearner()
        score, Xlindx, Xrindx = wlrner.train(X, Y)

        node = Node(wlearner = wlrner, score = score)
        node.split = wlrner.split
        
        if self.verbose:    
            print('Creating Left Child Node With {} Examples, and Right Node with {} Examples'
                  .format(len(Y[Xlindx]), len((Y[Xrindx]))))

        node.set_childs(self.build_tree(X[Xlindx], Y[Xlindx], depth-1),
                        self.build_tree(X[Xrindx], Y[Xrindx], depth-1))

        return node

    def test(self, X):
        ''' Test the trained classifiers on the given set of examples


            Input:
            ------
            X: [m x d] a data matrix of m d-dimensional test examples.

            Returns:
            -----------
                pclass: the predicted class for each example, i.e. to which it belongs
        '''

        nexamples, nfeatures = X.shape
        
        pclasses = self.predict(X)

        return np.array(pclasses)

    def predict(self, X):
        """
        Test the trained classifiers on the given example X


            Input:
            ------
            X: [m x d] a d-dimensional test example.

            Returns:
            -----------
                pclass: the predicted class for the given example, i.e. to which it belongs
        """
        X = np.atleast_2d(X)
        z = []

        for idx in range(X.shape[0]):
            z.append(self._predict(self.tree, X[idx, :]))

        return z

    def _predict(self, node, X):
        """
            recursively traverse the tree from root to child and return the child node label
            for the given example X
        """
        
        # If no rule is present return the class with highest probability
        if not node:
            return self.pdist[0][0]

        if node.isleaf():
            return node.pdistribution[0][0]
        if node.isless_than_eq(X):
            return self._predict(node.lchild, X)
        else:
            return self._predict(node.rchild, X)

    def __str__(self):
        """
            overloaded function used by print function for printing the current tree in a
            string format
        """
        str = '---------------------------------------------------'
        str += '\n A Decision Tree With Depth={}'.format(self.find_depth())
        str += self.__print(self.tree)
        str += '\n---------------------------------------------------'
        return str  # self.__print(self.tree)

    def __get_pdist(self, Y):
        keys, counts = np.unique(Y, return_counts=True)
        counts = counts / len(Y)
        dist = []
        for idx, key in enumerate(keys):
            dist.append((key, counts[idx]))
        dist.sort(key=lambda x: x[1], reverse=True)
        return dist

    def _print(self, node):
        """
                Recursive function traverse each node and extract each node information
                in a string and finally returns a single string for complete tree for printing purposes
        """
        if not node:
            return
        if node.isleaf():
            return node.get_str()

        string = node.get_str() + self._print(node.lchild)
        return string + node.get_str() + self._print(node.rchild)

    def find_depth(self):
        """
            returns the depth of the tree...
        """
        return self._find_depth(self.tree)

    def _find_depth(self, node):
        """
            recursively traverse the tree to the depth of the tree and return the depth...
        """
        if not node:
            return
        if node.isleaf():
            return 1
        else:
            return max(self._find_depth(node.lchild) or 0,
                       self._find_depth(node.rchild) or 0) + 1

    def __print(self, node, depth=0):
        """

        """
        ret = ""

        # Print right branch
        if node.rchild:
            ret += self.__print(node.rchild, depth + 1)

        # Print own value

        ret += "\n" + ("    "*depth) + node.get_str()

        # Print left branch
        if node.lchild:
            ret += self.__print(node.lchild, depth + 1)

        return ret
