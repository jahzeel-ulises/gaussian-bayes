# Gaussian Naive Bayes
This is a Gaussian Naive Bayes classifier, built completely on numpy.

Gaussian Naive Bayes (GNB) is a classification technique used in machine learning based on a probabilistic approach and Gaussian distribution.

For use it just clone the repository and install numpy, version used in the project is on the requirements.txt file.

```
git clone https://github.com/jahzeel-ulises/gaussian-bayes
pip install -r requirements.txt
```

## How to use it
This class basicaly follow the same workflow of sklean library.

```python
from GaussianNB import GaussianNB
import numpy as np

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
clf = GaussianNB()
clf.fit(X, Y)
print(clf.predict([[-0.8, -1]]))
```

## Class information

**Gaussian Naive Bayes (GaussianNB).**

*Parameters*

priors: dict
    Prior probabilities of the classes.

var_smoothing: float, default = 1e-9
    Portion of the largest variance of all features that is added to all
    variances to avoid numeric errors.

*Atributes*
self.epsilon_: float
    Value added to the variances.

self.feature_size_: int
    Number of features seen in the fit.
    
self.classes_: array-like
    Class labels.

self.classes_statics_: dict
    Dictionary with the variances and means of each class.

self.priors_ : dict
    Probability of each class.


**predict_proba(self,X)**:
Predict probabilities of be in one class.

*Parameters*
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    Training vector, where `n_samples` is the number of samples and
    `n_features` is the number of features.
        
*Returns*
ndarray: {array-like} of shape (n_samples,n_classes)
        Result vector, where 'n_samples' is the number of samples and 
        'n_classes' is the number of the classes.

**predict(self,X)**:
        
Return the class of the max-prob index.

*Parameters*
X:{array-like} of shape (n_features,)
    Array with the probabilities of each class.
        
*Returns*
ndarray:
    Array with the predicted classes.

**fit(self,X,y)**:
Fit the model according to the given training data.

*Parameters*
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    Training vector, where `n_samples` is the number of samples and
    `n_features` is the number of features.
y : array-like of shape (n_samples,)
    Target vector relative to X.
        
*Returns*
self:
    Fitted estimator. 

**score(self,X,y)**:
Calculates the accuracy of the model.
        
*Parameters*
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    Testing vector, where `n_samples` is the number of samples and
    `n_features` is the number of features.
y : array-like of shape (n_samples,)
    Target vector relative to X.
        
*Returns*
float:
    Accuracy value in [0,1]
       
