import numpy as np
import warnings

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
    
    if len(y_c.shape) != 1:
        warnings.warn("Data conversion warning, y shape expected (n_samples,) but vector column received")
        flag = True
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
        X_c = np.array(X)
        y_c = np.ravel(y)
        classes = np.unique(y)
        splited_data = dict()
        for _ in range(X.shape[0]):
            try:
                splited_data[y_c[_]] = np.hstack((splited_data[y_c[_]],X_c[_,:]))
            except:
                splited_data[y_c[_]] = X_c[_,:]
        return splited_data,classes

    def _calculate_classes_statics(self,splited_data:dict):
        """
        Generate descriptive statistics of each class.

        Parameters
        -----------------
        splited_data: dict
            Dictionary with the arrays separated by class.
        
        Returns
        -----------------
        dict:
            Dictionary with the descriptors of each class.

            Ex: {"1":{"mu":1,"sigma":0}}
        """

        classes_statics_ = dict()
        for class_ in self.classes_:
            mu = np.mean(splited_data[class_],axis=0)
            sigma = np.std(splited_data[class_],axis=0)
            classes_statics_[class_] = {"mu":mu,"sigma":sigma}
        
        return classes_statics_


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
        _validate_data(X,y)

        splited_data,self.classes_ = self._split_data(X,y)
        self.classes_statics_ = self._calculate_classes_statics(splited_data)
        return self
