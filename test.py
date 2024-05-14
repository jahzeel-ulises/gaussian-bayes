import numpy as np
import warnings

def _validate_data(X,y)->None:
    """
    Checks data input to be correct.
    
    Parameters
    -------------------
    X: Any
    y: Any

    Returns
    -------------------
    None

    Raises
    -------------------
    Exception: List, pandas data frame or numpy array expected
    Exception: 2D array like on X expected
    Exception: Unidimensional array on y expected
    Exception: Data conversion warning, y shape expected (n_samples,) but vector column received
    Exception: At least 2 classes needed
    Exception: X's n_sample != y'n n_sample

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
    
    #Checks X _samples same y n_samples
    y_c = np.ravel(y_c)
    if X_c.shape[0] != y_c.shape[0]:
        raise Exception("X's n_sample != y'n n_sample")

    
class GaussianNB():
    """
    """
    def __init__(self,priors:dict = None,var_smoothing=1e-9) -> None:
        self.priors_ = priors
        self.var_smoothing_ = var_smoothing
    
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
                splited_data[y_c[_]] = np.vstack((splited_data[y_c[_]],X_c[_,:]))
            except:
                splited_data[y_c[_]] = X_c[_,:]
        return splited_data,classes
    
    def _calculate_priors(self,y)->dict:
        """
        Calculates the priors probability, if set in constructor, validates it.

        Parameters
        -----------------
        y : array-like of shape (n_samples,)
            Target vector relative to X.
        
        Returns
        ----------------
        dict
            Python dictionary with priors probabilities.

        Raises
        --------------
        Exception: Dict in priors expected
        Exception: Class in priors not in training set
        """
        #If priors set in constructor validates priors are in y
        if self.priors_:
            if type(self.priors_) != dict:
                raise Exception("Dict in priors expected")

            for class_ in self.classes_:
                if class_ not in self.priors:
                    raise Exception("Class in priors not in training set")
            return self.priors_

        y_c = np.ravel(y)

        priors = dict()
        for class_ in self.classes_:
            priors[class_] = np.count_nonzero(y_c == class_)/y_c.shape[0]
        return priors
            

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
            mu = np.mean(splited_data[class_],axis=0,dtype=np.float64)
            sigma = np.var(splited_data[class_],axis=0,dtype=np.float64,ddof=1) + self.epsilon_
            classes_statics_[class_] = {"mu":mu,"sigma":sigma}
        
        return classes_statics_

    def _normal_dist(self,X ,mean,sigma):
        """
        Calculates the value of X entries in the normal distribution.

        Parameters
        -----------------
        X : {array-like} of shape (n_features,)
            Training vector `n_features` is the number of features.
        mean : {array-like} of shape (n_features,)
            Vector with mu values of each feature for a especific class.
        sigma : {array-like} of shape (n_features,)
            Vector with sigma values of each feature for a especific class.
        
        Returns
        -----------------
        numpy.ndarray
            Array with the normal distribution values of the entries.
        """
        return (1/((np.sqrt(sigma))*np.sqrt(2*np.pi))) * np.exp(-0.5*((X-mean)**2/np.sqrt(sigma)))
    
    def _sum_of_logs(self,X,mean,sigma):
        """
        Calculates the value of the sum of log of entries in X.

        Parameters
        -----------------
        X : {array-like} of shape (n_features,)
            Training vector `n_features` is the number of features.
        
        Returns
        -----------------
        numpy.ndarray
            Scalar variable with the sum of logs values.
        """
      
        logs = -0.5*np.log(sigma*2*np.pi) - 0.5*(((X - mean))**2/sigma)
        return np.sum(logs)
    
    def _product_of_likelihood(self,X):
        """
        Calculates the product of likelihood values.

        Parameters
        ----------------
        X : {array-like} of shape (n_features,)
            Training vector 'n_features' is the number of features.
        
        Returns
        ----------------
        numpy.ndarray
            Scalar variable with probability of be on certein class.
        """
        return np.prod(X)
    def predict_proba(self,X):
        """
        Predict probabilities of be in one class.

        Parameters
        -----------------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        
        Returns
        -----------------
        ndarray: {array-like} of shape (n_samples,n_classes)
               Result vector, where 'n_samples' is the number of samples and 
               'n_classes' is the number of the classes.
        """
        X_c = np.array(X)

        if X_c.shape[1] != self.feature_size_:
            raise Exception("X with shape (n_sample,n_features) expected")

        proba = np.empty((0,X_c.shape[0]))
        for class_ in self.classes_:
            mean = self.classes_statics_[class_]["mu"]
            sigma = self.classes_statics_[class_]["sigma"]
            norm_dist = np.apply_along_axis(self._normal_dist,1,X_c,mean,sigma)
            likelihood = np.apply_along_axis(self._product_of_likelihood,1,norm_dist)
            likelihood = self.priors_[class_]*likelihood
            proba = np.vstack((proba,likelihood))
        return proba.T
    
    def _get_classes(self,X):
        """
        Return the class of the max probability index.

        Parameters
        ---------------
        X: {array-like} with the probability of the class.

        Retuns
        ---------------
        object:
            Class of the 
        """
        return self.classes_[np.argmax(X)]

    def predict(self,X):
        """
        Return the class of the max-prob index.

        Parameters
        ----------------
        X:{array-like} of shape (n_features,)
            Array with the probabilities of each class.
        
        Returns
        ----------------
        ndarray:
            Array with the predicted classes.

        """
        X_c = np.array(X)

        if X_c.shape[1] != self.feature_size_:
            raise Exception("X with shape (n_sample,n_features) expected")
        
        pred = np.empty((0,X_c.shape[0]))
        for class_ in self.classes_:
            mean = self.classes_statics_[class_]["mu"]
            sigma = self.classes_statics_[class_]["sigma"]
            logs = np.apply_along_axis(self._sum_of_logs,1,X_c,mean,sigma)
            logs = logs + np.log(self.priors_[class_])
            pred = np.vstack((pred,logs))
        pred = pred.T

        return np.apply_along_axis(self._get_classes,1,pred)

    def score(self,X,y):
        """
        Calculates the accuracy of the model.
        
        Parameters
        -------------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Testing vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Target vector relative to X.
        
        Returns
        ------------
        float
            Accuracy value in [0,1]
        """
        y_pred = self.predict(X)
        pred_v_real = y == y_pred
        return np.count_nonzero(pred_v_real)/pred_v_real.shape[0]

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

        self.epsilon_ = self.var_smoothing_ * np.var(X, axis=0).max()
        self.feature_size_ = np.array(X).shape[1]
        splited_data,self.classes_ = self._split_data(X,y)
        self.classes_statics_ = self._calculate_classes_statics(splited_data)
        self.priors_ = self._calculate_priors(y)
        return self
