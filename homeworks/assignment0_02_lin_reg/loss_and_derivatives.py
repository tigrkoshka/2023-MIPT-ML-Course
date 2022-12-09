import numpy as np

class LossAndDerivatives:
    @staticmethod
    def mse(X, Y, w):
        """
        X : numpy array of shape (`n_observations`, `n_features`)
        Y : numpy array of shape (`n_observations`, `target_dimensionality`) or (`n_observations`,)
        w : numpy array of shape (`n_features`, `target_dimensionality`) or (`n_features`,)

        Return : float
            single number with MSE value of linear model (X.dot(w)) with no bias term
            on the selected dataset.
        
        Comment: If Y is two-dimensional, average the error over both dimensions.
        """

        return np.mean((X.dot(w) - Y) ** 2)

    @staticmethod
    def mae(X, Y, w):
        """
        X : numpy array of shape (`n_observations`, `n_features`)
        Y : numpy array of shape (`n_observations`, `target_dimensionality`) or (`n_observations`,)
        w : numpy array of shape (`n_features`, `target_dimensionality`) or (`n_features`,)
                
        Return: float
            single number with MAE value of linear model (X.dot(w)) with no bias term
            on the selected dataset.

        Comment: If Y is two-dimensional, average the error over both dimensions.
        """

<<<<<<< HEAD
        return np.mean(np.abs(X.dot(w) - Y))
=======
        # YOUR CODE HERE    
        return np.mean(np.abs(X @ w - Y))
>>>>>>> 0af341e (22f basic (#89))

    @staticmethod
    def l2_reg(w):
        """
        w : numpy array of shape (`n_features`, `target_dimensionality`) or (`n_features`,)

        Return: float
            single number with sum of squared elements of the weight matrix ( \sum_{ij} w_{ij}^2 )

        Computes the L2 regularization term for the weight matrix w.
        """
<<<<<<< HEAD

        return (w ** 2).sum()
=======
        
        # YOUR CODE HERE
        return (w.astype(float) ** 2).sum()
>>>>>>> 0af341e (22f basic (#89))

    @staticmethod
    def l1_reg(w):
        """
        w : numpy array of shape (`n_features`, `target_dimensionality`)

        Return : float
            single number with sum of the absolute values of the weight matrix ( \sum_{ij} |w_{ij}| )
        
        Computes the L1 regularization term for the weight matrix w.
        """

<<<<<<< HEAD
=======
        # YOUR CODE HERE
>>>>>>> 0af341e (22f basic (#89))
        return np.abs(w).sum()

    @staticmethod
    def no_reg(w):
        """
        Simply ignores the regularization
        """
        return 0.

    @staticmethod
    def mse_derivative(X, Y, w):
        """
        X : numpy array of shape (`n_observations`, `n_features`)
        Y : numpy array of shape (`n_observations`, `target_dimensionality`) or (`n_observations`,)
        w : numpy array of shape (`n_features`, `target_dimensionality`) or (`n_features`,)
        
        Return : numpy array of same shape as `w`

        Computes the MSE derivative for linear regression (X.dot(w)) with no bias term
        w.r.t. w weight matrix.
        
        Please mention, that in case `target_dimensionality` > 1 the error is averaged along this
        dimension as well, so you need to consider that fact in derivative implementation.
        """

<<<<<<< HEAD
        return 2 / Y.size * X.T.dot((X.dot(w) - Y))
=======
        # YOUR CODE HERE
        return 2 * X.T @  (X @ w - Y) / Y.size
>>>>>>> 0af341e (22f basic (#89))

    @staticmethod
    def mae_derivative(X, Y, w):
        """
        X : numpy array of shape (`n_observations`, `n_features`)
        Y : numpy array of shape (`n_observations`, `target_dimensionality`) or (`n_observations`,)
        w : numpy array of shape (`n_features`, `target_dimensionality`) or (`n_features`,)
        
        Return : numpy array of same shape as `w`

        Computes the MAE derivative for linear regression (X.dot(w)) with no bias term
        w.r.t. w weight matrix.
        
        Please mention, that in case `target_dimensionality` > 1 the error is averaged along this
        dimension as well, so you need to consider that fact in derivative implementation.
        """

<<<<<<< HEAD
        return 1 / Y.size * X.T.dot(np.sign(X.dot(w) - Y))
=======
        # YOUR CODE HERE
        return X.T @ np.sign(X @ w - Y) / Y.size
>>>>>>> 0af341e (22f basic (#89))

    @staticmethod
    def l2_reg_derivative(w):
        """
        w : numpy array of shape (`n_features`, `target_dimensionality`) or (`n_features`,)

        Return : numpy array of same shape as `w`

        Computes the L2 regularization term derivative w.r.t. the weight matrix w.
        """

<<<<<<< HEAD
=======
        # YOUR CODE HERE
>>>>>>> 0af341e (22f basic (#89))
        return 2 * w

    @staticmethod
    def l1_reg_derivative(w):
        """
        Y : numpy array of shape (`n_observations`, `target_dimensionality`) or (`n_observations`,)
        w : numpy array of shape (`n_features`, `target_dimensionality`) or (`n_features`,)

        Return : numpy array of same shape as `w`

        Computes the L1 regularization term derivative w.r.t. the weight matrix w.
        """
<<<<<<< HEAD

        return np.sign(w)
=======
        # YOUR CODE HERE
        return np.sign(w).astype(float)
>>>>>>> 0af341e (22f basic (#89))

    @staticmethod
    def no_reg_derivative(w):
        """
        Simply ignores the derivative
        """
        return np.zeros_like(w)
