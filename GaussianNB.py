import numpy as np

def _validate_data(X,y)->bool:
    """
    Checks data input to be correct.
    
    Parameters
    -------------------
    X: Any
    y: Any

    Returns
    -------------------
    bool

    Raises
    -------------------
    Exception: List, pandas data frame or numpy array expected
    """
    
    #Check the data to be list, pandas data frame or numpy array
    try:
         X_c = np.array(X)
         y_c = np.array(y)
    except:
         raise Exception("List, pandas data frame or numpy array expected")

    #Checks data shape be correct
    if len(X_c.shape) != 2:
        raise Exception("2D array like on X expected")
    
    if len(y_c.shape) > 2:
        raise Exception("Unidimensional array on y expected")
    
    if len(y_c.shape) != 1 and (y_c.shape[0] != 1 and y_c.shape[1] != 1):
        raise Exception("Unidimensional array on y expected")
    
    #Checks number of classes(minimum 2)
    if len(np.unique(y_c)) < 2:
        raise Exception("At least 2 classes needed")
    
class GaussianNB():
    """
    """
    def __init__(self) -> None:
        pass

    
    def _split_data(self,X,y)->dict:
        """
        Splits X on differents arrays grouped by the class.

        Parameters
        ---------------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Target vector relative to X.

        Retuns
        --------------
        dict:
            Dictionary with matrix group by classes.
        """
        self.classes_ = np.unique(y)
        y_c = np.ravel(y)
        matrix = dict()
        for _ in range(X.shape[0]):
            try:
                matrix[y[_]] = np.hstack((matrix[y[_]],X[_,:]))
            except:
                matrix[y[_]] = X[_,:]
        return matrix


    def fit(self,X,y):
        """
        Fit the model according to the given training data.

        Parameters
        -------------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Target vector relative to X.
        
        Returns
        -------------
        self:
            Fitted estimator.
        """
        return self
