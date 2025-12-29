# import numpy as np
import cupy as cp
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class multinomial_logistic_regression(BaseEstimator, ClassifierMixin):
    ''''
    multinmial multilayered logistic regression classifier
    supports GPU processing and sklearn API
    '''

    def __init__(self, eta=0.1, max_iter=2000, batch_size=None, random_state=None, decay=1e-4,
                 early_stopping=False, patience=30, tol=1e-6, validation_fraction=0.1,
                 hidden_layers=[100], verbose=0):
        self.eta = eta
        self.max_iter = max_iter
        self.random_state = random_state
        self.decay = decay
        self.batch_size = batch_size
        self.patience = patience
        self.tol = tol
        self.validation_fraction = validation_fraction
        self.early_stopping = early_stopping if validation_fraction>0 else False
        self.hidden_layers = hidden_layers
        self.verbose = verbose
        self.weights_ = []
        self.biases_ = []
        self.classes_ = []
        self.losses = []
        self.val_losses = []

    def _sigmoid(self, z):
        return 1. / (1. + cp.exp(-cp.clip(z, -250, 250)))

    def _sigmoid_derivative(self, a):
        # derivative of a with respect to its z (element-wise derivation)
        return a * (1. - a)

    def _forward(self, X):
        yhats = [X]
        zs = []
        cur = X

        if not self.weights_:
            raise RuntimeError("model is not fitted yet. call 'fit'.")

        for i in range(len(self.weights_) - 1): #n_layers -1
            z = cur @ self.weights_[i] + self.biases_[i]
            a = self._sigmoid(z) #activation on hidden is just sigmoid
            zs.append(z) #node net input
            yhats.append(a) # node yhat
            cur = a

        z_final = cur @ self.weights_[-1] + self.biases_[-1]
        a_final = self.activation(z_final) #on final layer, activation is softmax
        zs.append(z_final)
        yhats.append(a_final)

        return zs, yhats

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        self.classes_ = unique_labels(y) #sort labels and store indices
        n_classes = len(self.classes_)
        y_mapped = np.searchsorted(self.classes_, y) #map labels lexicographical as indexes

        n_samples, n_features = X.shape
        # GPU computations start:
        X = cp.array(X)
        y_enc = self._one_hot(cp.array(y_mapped), len(self.classes_))
        rgen = cp.random.RandomState(self.random_state)

        layer_sizes = [n_features] + self.hidden_layers + [n_classes]

        # --- Generate w0, b0 ---
        self.weights_ = []
        self.biases_ = []
        for i in range(len(layer_sizes) - 1):
            std_dev = cp.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i + 1])) # variance = (1+1) / N
            w = rgen.normal(loc=0.0, scale=std_dev, size=(layer_sizes[i], layer_sizes[i + 1]))
            b = cp.zeros(layer_sizes[i + 1])
            self.weights_.append(w)
            self.biases_.append(b)
        # parse to np via list comprehension :
        best_weights = [w.copy() for w in self.weights_]
        best_biases = [b.copy() for b in self.biases_]
        best_val_loss = float(cp.inf)
        epochs_no_improve = 0
        self.hist_ = []
        self.losses = []

        # --- Val Split ---
        if self.validation_fraction and self.validation_fraction > 0:
            val_n = int(self.validation_fraction * n_samples)
            perm = rgen.permutation(n_samples)
            X_val, X = X[perm][:val_n], X[perm][val_n:]
            y_val_enc, y_enc = y_enc[perm][:val_n], y_enc[perm][val_n:]
            self.val_losses = []
        else:
            X_val = y_val_enc = None
        if self.batch_size is None:
            self.batch_size = 1

        for epoch in range(self.max_iter):
            idx = rgen.permutation(n_samples).astype(int)
            X_sh, y_sh = X[idx], y_enc[idx]

            for l in range(0, n_samples, self.batch_size):
                r = l + self.batch_size
                X_, y_ = X_sh[l:r], y_sh[l:r]
                zs, yhats = self._forward(X_)

                # --- Output Layer ---
                Gw = []
                Gb = []
                d = yhats[-1] - y_
                dw = yhats[-2].T @ d / X_.shape[0]
                db = d.mean(axis=0)
                Gw.append(dw)
                Gb.append(db)

                # --- Hidden Layers (backwards) ---
                n_layers = len(self.weights_)
                for i in range(n_layers - 2, -1, -1): # from N-2 to 0
                    err = d @ self.weights_[i + 1].T # error from this node on the (node output)
                    d = err * self._sigmoid_derivative(yhats[i + 1]) # element-wise error = err * f'(yhat(outpot))

                    dw = yhats[i].T @ d / X_.shape[0]
                    db = d.mean(axis=0)
                    # push-front the layer parameters
                    Gw.insert(0, dw)
                    Gb.insert(0, db)


                lr = self.eta / (1 + self.decay * epoch) # lambda (decaying eta)
                for i in range(len(self.weights_)): # GD on each layer
                    self.weights_[i] -= lr * Gw[i]
                    self.biases_[i] -= lr * Gb[i]

                loss = self.multinomial_cross_entropy(cp.argmax(y_, axis=1), yhats[-1])
            self.losses.append(loss)


            # --- Validation & Early Stopping ---
            if X_val is not None and (epoch % 50 == 0 or epoch == self.max_iter - 1):
                val_proba = self._forward(X_val)[1][-1] # second item in tuple(zs, yhats), get last layer.
                y_val_idx = cp.argmax(y_val_enc, axis=1)
                v_loss = self.multinomial_cross_entropy(y_val_idx, val_proba)
                self.val_losses.append(v_loss)

                if v_loss < best_val_loss:
                    best_val_loss = v_loss
                    self.best_epoch_ = epoch + 1
                    self.hist_.append([epoch, v_loss])

                    if self.early_stopping:
                        epochs_no_improve = 0
                        #parse to np via list comprehension
                        best_weights = [w.copy() for w in self.weights_]
                        best_biases = [b.copy() for b in self.biases_]
                else:
                    if self.early_stopping:
                        epochs_no_improve += 1

                if self.early_stopping and epochs_no_improve >= self.patience:
                    self.weights_ = best_weights
                    self.biases_ = best_biases
                    print(f"early stopping at epoch {epoch + 1}")
                    break

            if self.verbose>0:
                if epoch % self.verbose == 0 or epoch == self.max_iter -1:
                    print(f'epoch {epoch:4d} training loss {loss:.4f}', end='')
                    if hasattr(self, 'val_losses') and self.val_losses:
                        print(f' - val loss {self.val_losses[-1]:.4f}')
                    else:
                        print('')

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        X = cp.array(X) # do in GPU
        _, proba = self._forward(X)
        return cp.argmax(proba[-1], axis=1).get() # get output layer prediction from cupy

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        X=cp.array(X) # do in GPU
        yhats = self._forward(X)[1] # returns tuple(zs,yhats)
        return cp.asnumpy(yhats[-1])  # get output layer probabilities

    def activation(self, z):
        z_max = cp.max(z, axis=1, keepdims=True)
        exp_z = cp.exp(z - z_max) # z-zmax to regularize softmax large sz.
        sum_exp = cp.sum(exp_z, axis=1, keepdims=True)
        return exp_z / sum_exp

    def multinomial_cross_entropy(self, y_true, y_pred):
        y_pred = cp.clip(y_pred, 1e-15, 1 - 1e-15)
        confidence = y_pred[cp.arange(y_pred.shape[0]), y_true] # the max proba class, and the rest mul by 0
        return -cp.mean(cp.log(confidence)) # -mean(log(conf)) mean substitutes dividing by x.shape[0] for normalization

    def _one_hot(self, y, n_classes):
        return cp.eye(n_classes)[y] # do identity, and pick row[i] if y=i