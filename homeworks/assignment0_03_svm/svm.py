from sklearn.base import BaseEstimator, ClassifierMixin

import numpy as np
import torch
import torch.optim as optim


def rbf(x_1, x_2, sigma=1.):
    """Computes rbf kernel for batches of objects

    Args:
        x_1: torch.tensor shaped `(#samples_1, #features)` of type torch.float32
        x_2: torch.tensor shaped `(#samples_1, #features)` of type torch.float32
    Returns:
        kernel function values for all pairs of samples from x_1 and x_2
        torch.tensor of type torch.float32 shaped `(#samples_1, #samples_2)`
<<<<<<< HEAD
    """
    x_1_2 = (x_1 ** 2).sum(axis=1).reshape(x_1.shape[0], 1)
    x_2_2 = (x_2 ** 2).sum(axis=1).reshape(1, x_2.shape[0])
    x_1_x_2 = x_1 @ x_2.T
    distances = torch.exp(-(x_1_2 + x_2_2 - 2 * x_1_x_2) / (2 * sigma ** 2))
=======
    '''
    dists = np.sqrt((X**2).sum(axis=1)[:, np.newaxis] + (self.X_train**2).sum(axis=1) - 2 * X.dot(self.X_train.T))
    distances = np.exp(-sigma * distances)
>>>>>>> 0af341e (22f basic (#89))
    return torch.Tensor(distances).type(torch.float32)


def hinge_loss(scores, labels):
    """Mean loss for batch of objects
    """
    assert len(scores.shape) == 1
    assert len(labels.shape) == 1
<<<<<<< HEAD
    return torch.mean(torch.max(torch.zeros(scores.shape[0]), 1 - scores * labels))
=======
    return torch.mean(torch.clamp(1 - scores * labels, 0))
>>>>>>> 0af341e (22f basic (#89))


class SVM(BaseEstimator, ClassifierMixin):
    @staticmethod
    def linear(x_1, x_2):
        """Computes linear kernel for batches of objects

        Args:
            x_1: torch.tensor shaped `(#samples_1, #features)` of type torch.float32
            x_2: torch.tensor shaped `(#samples_1, #features)` of type torch.float32
        Returns:
            kernel function values for all pairs of samples from x_1 and x_2
            torch.tensor shaped `(#samples_1, #samples_2)` of type torch.float32
<<<<<<< HEAD
        """
        return x_1 @ x_2.T

=======
        '''
        return x_1 @ x_2.T
    
>>>>>>> 0af341e (22f basic (#89))
    def __init__(
            self,
            lr: float = 1e-3,
            epochs: int = 2,
            batch_size: int = 64,
            lmbd: float = 1e-4,
            kernel_function=None,
            verbose: bool = False,
    ):
        self.X = None
        self.bias = None
        self.betas = None
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.lmbd = lmbd
        self.kernel_function = kernel_function or SVM.linear
        self.verbose = verbose
        self.fitted = False

    def __repr__(self):
        return 'SVM model, fitted: {self.fitted}'

    def fit(self, X, Y):
        assert (np.abs(Y) == 1).all()
        n_obj = len(X)
        X, Y = torch.FloatTensor(X), torch.FloatTensor(Y)
        K = self.kernel_function(X, X).float()

        self.betas = torch.full((n_obj, 1), fill_value=0.001, dtype=X.dtype, requires_grad=True)
        self.bias = torch.zeros(1, requires_grad=True)  # I've also add bias to the model

        optimizer = optim.SGD((self.betas, self.bias), lr=self.lr)
        for epoch in range(self.epochs):
            perm = torch.randperm(n_obj)  # Generate a set of random numbers of length: sample size
            sum_loss = 0.  # Loss for each epoch
            for i in range(0, n_obj, self.batch_size):
                batch_inds = perm[i:i + self.batch_size]
                y_batch = Y[batch_inds]  # Pick the correlating class
                k_batch = K[batch_inds]
<<<<<<< HEAD

                optimizer.zero_grad()  # Manually zero the gradient buffers of the optimizer

                # get the matrix product using SVM parameters: self.betas and self.bias
                preds = k_batch @ self.betas + self.bias
=======
                
                optimizer.zero_grad()     # Manually zero the gradient buffers of the optimizer
                
                preds = k_batch @ self.betas + self.bias ### YOUR CODE HERE # get the matrix product using SVM parameters: self.betas and self.bias
>>>>>>> 0af341e (22f basic (#89))
                preds = preds.flatten()
                loss = self.lmbd * self.betas[batch_inds].T @ k_batch @ self.betas + hinge_loss(preds, y_batch)
                loss.backward()  # Backpropagation
                optimizer.step()  # Optimize and adjust weights

                sum_loss += loss.item()  # Add the loss

            if self.verbose:
                print("Epoch " + str(epoch) + ", Loss: " + str(sum_loss / self.batch_size))

        self.X = X
        self.fitted = True
        return self

    def predict_scores(self, batch):
        with torch.no_grad():
            batch = torch.from_numpy(batch).float()
            K = self.kernel_function(batch, self.X)
            # compute the margin values for every object in the batch
<<<<<<< HEAD
            return (K @ self.betas + self.bias).flatten()
=======
            return (K @ self.betas + self.bias).flatten()### YOUR CODE HERE
>>>>>>> 0af341e (22f basic (#89))

    def predict(self, batch):
        scores = self.predict_scores(batch)
        answers = np.full(len(batch), -1, dtype=np.int64)
        answers[scores > 0] = 1
        return answers
