# import numpy as np
import os
import cupy as cp
import numpy as np
import cupyx.scipy.signal as signal
from sklearn.base import BaseEstimator, ClassifierMixin
import inspect
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef
from sklearn.utils.multiclass import unique_labels
from typing import Annotated, Tuple, List, Union
Tensor = cp.ndarray


class BaseNeuralNet(BaseEstimator, ClassifierMixin):
    def __init__(self, eta=0.1, max_iter=2000, batch_size=32, random_state=None, clip_value=5 ,decay=1e-4,
                 early_stopping=False, patience=30, tol=1e-6, validation_fraction=0.1, val_jump =1,
                 verbose=0, activation='relu', task = 'classification', metric ='accuracy'):
        self.eta = eta
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.random_state = random_state
        self.decay = decay
        self.early_stopping = early_stopping
        self.patience = patience
        self.verbose = verbose
        self.activation = activation
        self.tol = tol
        self.validation_fraction = validation_fraction
        self .val_jump = val_jump
        self.weights_ = []
        self.biases_ = []
        self.losses_ = []
        self.val_losses_ = []
        self.val_scores_ =[]
        self.scores_ =[]
        self.best_val_loss_ = float(cp.inf)
        self.epochs_no_improve_ = 0
        self.task = task
        self.metric = metric
        self.clip_value = clip_value

        self._init_activation_functions()

    def get_params(self, deep=True):
        """
        Merge child and parent parameters to be fully visible to scikit-learn.
        """
        ch_params = super().get_params(deep)
        base_sig = inspect.signature(BaseNeuralNet.__init__)

        for name, param in base_sig.parameters.items():
            if name == 'self' or param.kind == param.VAR_KEYWORD: continue
            # add to dict if passed via **kwargs:
            if hasattr(self, name):
                ch_params[name] = getattr(self, name)

        return ch_params


    def _init_activation_functions(self):
        """
        Derivatives are calculated w.r.t 'a' (output).
        """
        self.activation_functions_ = {
            'relu': (self._relu, self._relu_deriv),
            'sigmoid': (self._sigmoid, self._sigmoid_deriv),
            'tanh': (self._tanh, self._tanh_deriv),
            'leaky_relu': (self._leaky_relu, self._leaky_relu_deriv),
            'softplus': (self._softplus, self._softplus_deriv),
            'identity': (lambda z: z, lambda a: cp.ones_like(a)),
            'softmax': (self._softmax, None)
        }

    def _relu(self, z): return cp.maximum(0, z)
    def _relu_deriv(self, a): return cp.where(a > 0, 1, 0)
    def _leaky_relu(self, z): return cp.where(z > 0, z, 0.01 * z)
    def _leaky_relu_deriv(self, a): return cp.where(a > 0, 1, 0.01)
    def _sigmoid(self, z): return 1 / (1 + cp.exp(-cp.clip(z, -250, 250)))
    def _sigmoid_deriv(self, a): return a * (1 - a)
    def _tanh(self, z): return cp.tanh(z)
    def _tanh_deriv(self, a): return 1 - a ** 2
    def _softplus(self, z): return cp.log(1 + cp.exp(cp.clip(z, -250, 250)))
    def _softplus_deriv(self, a): return 1 - cp.exp(-a)  # 1 - 1/e^a = 1 - 1/(1+e^z) = sigmoid(z)
    def _softmax(self, z):
        ex = cp.exp(z - cp.max(z, axis=1, keepdims=True))
        return ex / cp.sum(ex, axis=1, keepdims=True)

    def _activate(self, z, name):
        if name not in self.activation_functions_: raise ValueError(f"unknown activation: {name}")
        return self.activation_functions_[name][0](z)

    def _deriv(self, a, name):
        if name not in self.activation_functions_: raise ValueError(f"unknown activation: {name}")
        if self.activation_functions_[name][1] is None:
            # Softmax uses "log‑sum‑exp" trick; then propagate back and calculates (yhat - y)
            # for each component without computing the Jacobian matrix (off-diag contributions vanish on
            # one-hot encoded y); the same update trick also applies to 0.5 * MSE/identity for regression
            # tasks <polymorphic consistent> because of the canonical link between logit and identity
            raise ValueError(f"derivative for {name} is handled in Loss/Jacobian.")
        return self.activation_functions_[name][1](a)

    def _initialize_architecture(self, n_features, n_classes):
        raise NotImplementedError("subclasses must implement _initialize_architecture()")

    def _forward(self, X):
        raise NotImplementedError("subclasses must implement _forward()")

    def _get_gradients(self, X, y, zs, acts):
        raise NotImplementedError("subclasses must implement _get_gradients()")

    def _one_hot(self, y, n_classes):
        return cp.eye(n_classes)[y]

    def _score_metric(self, y_true, y_pred):
        if y_true.ndim > 1:
            y_t = cp.argmax(y_true, axis=1)
        else:
            y_t = y_true
        y_p = cp.argmax(y_pred, axis=1)

        if self.metric == 'accuracy':
            return cp.mean(y_t == y_p)

        n_classes = self.n_classes_
        true = (y_t[:, None] == cp.arange(n_classes))
        pred = (y_p[:, None] == cp.arange(n_classes))

        tp = cp.sum(true & pred, axis=0)
        fp = cp.sum((~true) & pred, axis=0)
        fn = cp.sum(true & (~pred), axis=0)
        epsilon = 1e-7

        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1 = 2 * (precision * recall) / (precision + recall + epsilon)

        w = cp.sum(true, axis=0)
        tot_w = cp.sum(w)

        if self.metric == 'precision':
            return cp.sum(precision * w) / tot_w
        elif self.metric == 'recall':
            return cp.sum(recall * w) / tot_w
        elif self.metric == 'f1':
            return cp.sum(f1 * w) / tot_w
        return 0.0


    def _setup_task(self, y):
        if self.task == 'classification':
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)
            y_mapped = np.searchsorted(self.classes_, y)
            y_proc = self._one_hot(y_mapped, self.n_classes_)

            final_act = 'softmax'
            self._loss_func = self.multinomial_cross_entropy

        elif self.task == 'regression':
            # make y shape [N, Outputs]
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            y_proc = cp.array(y)
            self.n_classes_ = y.shape[1]  # n_outputs for regression

            final_act = 'identity'
            self._loss_func = self.mean_squared_error

        else:
            raise ValueError(f"Unknown task: {self.task}")

        return y_proc, final_act
    def multinomial_cross_entropy(self, y_true, y_pred):
        # y_true is one_hot_encoded
        eps = 1e-15
        y_pred = cp.clip(y_pred, eps, 1 - eps)
        return -cp.mean(cp.sum(y_true * cp.log(y_pred), axis=1)).get()

    def mean_squared_error(self, y_true, y_pred):
        # 0.5 * MSE makes the gradient exactly (pred - y)
        return 0.5 * cp.mean((y_true - y_pred) ** 2).get()

    def _validate_epoch(self, X, y, epoch):
        proba = self._forward(X)[1][-1]
        loss = self._loss_func(y, proba)
        self.val_losses_.append(loss)
        score = self._score_metric(y, proba)

        if not hasattr(self, 'val_scores_'): self.val_scores_ = []
        self.val_scores_.append(score)

        if loss < self.best_val_loss_:
            self.best_val_loss_ = loss
            self.best_epoch_ = epoch + 1
            if self.early_stopping:
                self.epochs_no_improve_ = 0
                self.best_weights_ = [w.copy() for w in self.weights_]
                self.best_biases_ = [b.copy() for b in self.biases_]
        else:
            if self.early_stopping: self.epochs_no_improve_ += 1

        if self.early_stopping and self.epochs_no_improve_ >= self.patience:
            self.weights_ = self.best_weights_
            self.biases_ = self.best_biases_
            print(f"Early stopping at epoch {epoch + 1}")
            return True
        return False

    def _process_batch(self, X, y, epoch):
        #Forward
        zs, acts = self._forward(X)
        # Backward
        Gw, Gb = self._get_gradients(X, y, zs, acts)
        # Update
        lr = self.eta / (1 + self.decay * epoch)
        for i in range(len(self.weights_)):
            gw = cp.clip(Gw[i], -self.clip_value, self.clip_value)
            gb = cp.clip(Gb[i], -self.clip_value, self.clip_value)
            self.weights_[i] -= lr * gw
            self.biases_[i] -= lr * gb
        return self._loss_func(y, acts[-1]), acts[-1]

    def fit(self, XX, yy):
        if self.task == 'classification':
            XX, yy = check_X_y(XX, yy)
        else:
            XX = check_array(XX)
            yy = np.array(yy)
        XX = cp.array(XX)
        y_proc, self.final_activation_ = self._setup_task(yy)

        self.n_samples_, self.n_features_ = XX.shape
        self._initialize_architecture(self.n_features_, y_proc.shape[1])

        rgen = cp.random.RandomState(self.random_state)
        if self.validation_fraction and self.validation_fraction > 0:
            val_n = int(self.validation_fraction * self.n_samples_)
            perm = rgen.permutation(self.n_samples_)
            X_val, XX = XX[perm][:val_n], XX[perm][val_n:]
            y_val_enc, y_proc = y_proc[perm][:val_n], y_proc[perm][val_n:]
            self.n_samples_ = XX.shape[0]
            self.val_losses_ = []
        else:
            X_val = y_val_enc = None

        if self.batch_size is None: self.batch_size = 32
        self.activation = str.lower(self.activation)

        for epoch in range(self.max_iter):
            idx = rgen.permutation(self.n_samples_).astype(int)
            X_sh, y_sh = XX[idx], y_proc[idx]
            for l in range(0, self.n_samples_, self.batch_size):
                r = l + self.batch_size
                X, y = X_sh[l:r], y_sh[l:r]

                loss, pred = self._process_batch(X, y, epoch)
                score = self._score_metric(y, pred)
            self.losses_.append(loss)
            self.scores_.append(score)
            if X_val is not None and (epoch % self.val_jump == 0 or epoch == self.max_iter - 1):
                if self._validate_epoch(X_val, y_val_enc, epoch): break

            if self.verbose > 0 and (epoch % self.verbose == 0 or epoch == self.max_iter - 1):
                print(f'epoch {epoch:4d} loss {loss:.5f}', end='')
                if self.val_losses_:
                    print(f' - val_loss {self.val_losses_[-1]:.5f}')
                else:
                    print('')
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)

        n_samples = X.shape[0]
        results = []
        for l in range(0, n_samples, self.batch_size):
            r = l + self.batch_size
            X_batch = cp.array(X[l:r])
            acts = self._forward(X_batch)[1]
            batch_probs = cp.asnumpy(acts[-1])
            results.append(batch_probs)
            del X_batch
            del acts
            cp.get_default_memory_pool().free_all_blocks()
        return np.concatenate(results, axis=0)

    def predict(self, X):
        scores = self.predict_proba(X)
        if self.task == 'regression':
            return scores
        return np.argmax(scores, axis=1)


class NeuralNetwork(BaseNeuralNet):
    def __init__(self, layers=[100], **kwargs):
        super().__init__(**kwargs)
        self.layers = layers

    def _initialize_architecture(self, n_features, n_outputs):
        """
        initializes MLP specific weights (He Init)
        """
        rgen = cp.random.RandomState(self.random_state)
        self.layer_sizes_ = [n_features] + self.layers + [n_outputs]
        self.n_layers_ = len(self.layer_sizes_)
        self.weights_ = []
        self.biases_ = []

        for i in range(self.n_layers_ - 1):
            if(self.activation == 'relu'):
                # He Initialization
                std = cp.sqrt(2.0 / self.layer_sizes_[i])
            else:
                # Xavier Initialization
                std = cp.sqrt(2.0 / (self.layer_sizes_[i]+ self.layer_sizes_[i+1]))
            w = rgen.normal(0.0, scale=std, size=(self.layer_sizes_[i], self.layer_sizes_[i + 1]))
            b = cp.zeros((1, self.layer_sizes_[i + 1]))
            self.weights_.append(w)
            self.biases_.append(b)

        self.best_weights_ = [w.copy() for w in self.weights_]
        self.best_biases_ = [b.copy() for b in self.biases_]


    def _forward(self, X):
        zs = []
        acts = [X]
        cur = X

        for i in range(len(self.weights_)):
            z = cur @ self.weights_[i] + self.biases_[i]
            zs.append(z)
            if i == len(self.weights_) - 1:
                a = self._activate(z, self.final_activation_)
            else:
                a = self._activate(z, self.activation)
            acts.append(a)
            cur = a

        return zs, acts

    def _get_gradients(self, X, y, zs, acts):
        """
        calculates gradients for MLP (Backprop)
        """
        Gw = [None] * len(self.weights_)
        Gb = [None] * len(self.biases_)

        d= acts[-1] - y
        # classification(ce + softmax)/regression(mse + I) implements the Jacobian trick,
        # mirroring canonical link property [logit - identity] yielding polymorphic consistent gradient

        Gw[-1] = acts[-2].T @ d / X.shape[0]
        Gb[-1] = cp.mean(d, axis=0, keepdims=True)

        for i in range(len(self.weights_) - 2, -1, -1):
            d = d @ self.weights_[i + 1].T * self._deriv(acts[i + 1], self.activation)
            Gw[i] = acts[i].T @ d / X.shape[0]
            Gb[i] = cp.mean(d, axis=0, keepdims=True)

        return Gw, Gb