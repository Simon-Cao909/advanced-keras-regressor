from sklearn.base import ClassifierMixin
from .estimator import SKGraphEstimator
from sklearn.metrics import accuracy_score

class SKGraphClassifier(SKGraphEstimator, ClassifierMixin):
    def score(self,X,y):
        '''
        Scores the model based on how it performs on given data

        :param X (array-like): The features of shape (n_samples, ...)
        :param y (array-like): The labels of shape (n_samples, ...) or (n_samples,)

        :return (float or ndarray of floats or None): The accuracy score or ndarray of scores
        '''
        if len(self.output_shape_) <= 2:
            return accuracy_score(y, self.predict(X))
        else:
            raise ValueError("Scoring is only defined for vector-valued outputs")