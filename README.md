# Extensible MLP: A First Principles Deep Learning Framework

> "I didn't want to just call model.fit(). I wanted to own the logic." [**View The Code**](neural_net.py)

Extensible MLP is a GPU-accelerated, Scikit-Learn compatible neural network framework built entirely from scratch using CuPy. It was born out of a desire to peel back the "black box" of modern libraries and implement the complex math of backpropagation manually.

This isn't just a reimplementation of existing tools; it is a journal of understanding how neural networks actually work, from the matrix calculus up to production-level memory management.


## The Philosophy

This project started with a strict set of rules for myself:

1.  **No generic AI help:** I avoided asking LLMs to "write a neural net" for me. I wanted the struggle.
2.  **First principles:** If I couldn't derive the math (gradients, Jacobians) on paper, I wouldn't code it.
3.  **Analyze, don't guess:** When the loss stalled, I spent days analyzing *why* rather than blindly copy-pasting fixes.
4.  **Scale:** It had to work on real data (MNIST), which meant handling GPU memory and matrix efficiency properly.

## The "Triple BAM" Moment: Deriving the Math
<img width="1600" height="780" alt="image" src="https://github.com/user-attachments/assets/94f5471b-4e90-4bf3-bbcb-6d14b7c36e36" />

The core breakthrough of this engine came when I was struggling to implement the backpropagation for different tasks. I was writing separate, complex derivatives for Regression (MSE) and Classification (Cross-Entropy).

I went back to the whiteboard and derived the gradients manually. I realized that for Generalized Linear Models (GLMs), if you pair the right activation function with the right loss function (the "Canonical Link"), the complex Jacobian matrix of the activation function cancels out perfectly with the derivative of the loss function.

The result was a unified, polymorphic gradient equation:

$$\\frac{\\partial L}{\\partial z} = \\hat{y} - y$$

* **Regression:** Identity activation + MSE loss.
* **Classification:** Softmax activation + Cross-Entropy loss.

In both cases, the gradient is simply the difference between the prediction and the target. This realization allowed me to write a single backpropagation engine that adapts to the task automatically.

## Engineering Highlights

### 1. The Polymorphic Architecture
Initially, I thought I needed separate classes for Regressor and Classifier. Realizing the math was identical allowed me to use a Strategy Pattern. The network reconfigures its final layer and loss function at runtime based on the target data.

I implemented this in [`_setup_task`](neural_net.py#L148), which inspects the target `y` and configures the engine dynamically:

```python
def _setup_task(self, y):
    if self.task == 'classification':
        self.classes_ = np.unique(y)
        # ... processing logic ...
        final_act = 'softmax'
        self._loss_func = self.multinomial_cross_entropy

    elif self.task == 'regression':
        # ... processing logic ...
        final_act = 'identity'
        self._loss_func = self.mean_squared_error
    
    return y_proc, final_act
```

### 2. Solving the Stalling Loss (Initialization)
Early in the project, my loss function would constantly stall at ~2.3. After days of debugging, I realized my weight initialization was ignoring the properties of my activation functions.

I implemented dynamic initialization strategies based on the papers by Glorot and He:
* **ReLU/Leaky ReLU:** Uses He Initialization.
* **Sigmoid/Tanh/Softmax:** Uses Xavier (Glorot) Initialization.

[View Code](neural_net.py#L309)
```python
# From _initialize_architecture
if self.activation == 'relu':
    # He Initialization (Variance = 2/n_in)
    std = cp.sqrt(2.0 / self.layer_sizes_[i])
else:
    # Xavier Initialization (Variance = 2/(n_in + n_out))
    std = cp.sqrt(2.0 / (self.layer_sizes_[i] + self.layer_sizes_[i+1]))
    
w = rgen.normal(0.0, scale=std, size=(n_in, n_out))
```

### 3. GPU Memory Management ("Scar Tissue")
Moving from toy datasets to MNIST on a GPU introduced immediate crashes (OOM errors). I had to implement "scar tissue" logic; code written specifically to handle the reality of hardware limits.

In [`predict_proba`](neural_net.py#L275), I enforce a hard cleanup after every batch to ensure VRAM never leaks, allowing the model to inference on datasets infinitely larger than the GPU memory.

```python
for l in range(0, n_samples, self.batch_size):
    # ... forward pass ...
    batch_probs = cp.asnumpy(acts[-1]) # Move to CPU immediately
    results.append(batch_probs)
    
    # Explicit cleanup to prevent VRAM accumulation
    del X_batch
    del acts
    cp.get_default_memory_pool().free_all_blocks()
```

### 4. Scikit-Learn Integration
To make the tool usable in real workflows, I implemented the full Scikit-Learn API. I used Python's `inspect` module to dynamically fetch parameters from the parent class, allowing `GridSearchCV` and `Pipeline` to "see" my custom hyperparameters without writing redundant boilerplate code in every child class.

[View Code](neural_net.py#L43)
```python
def get_params(self, deep=True):
    ch_params = super().get_params(deep)
    base_sig = inspect.signature(BaseNeuralNet.__init__)

    for name, param in base_sig.parameters.items():
        if name == 'self' or param.kind == param.VAR_KEYWORD: continue
        if hasattr(self, name):
            ch_params[name] = getattr(self, name)
    return ch_params
```

## Features

* **GPU Acceleration:** Built on CuPy for matrix operations.
* **Polymorphic Backprop:** Unified gradient engine for regression and classification.
* **Metrics Bridge:** Calculates metrics (F1, Precision, Recall) purely on the GPU to avoid CPU synchronization stalls.
* **Safety:** Automatic batching and memory cleanup.
* **Extensibility for Complex Architectures (CNN/RNN Ready):** I didn't just build an MLP; I built a generic trainer. By decoupling the engine (`BaseNeuralNet`) from the architecture (`NeuralNetwork`), I made it so you only need to override two methods; `_initialize_architecture` and `_forward`; to implement a CNN or RNN. The entire training loop, validation logic, and backprop engine stay exactly the same.
* **Advanced Scikit-Learn Compatibility:** Most custom implementations break when you try to use them with `GridSearchCV` or `Pipeline` because they don't expose parameters correctly. I used Python's `inspect` module to dynamically fetch parameters from the parent class, making this model a true drop-in replacement for any Sklearn estimator. You can plug it straight into a Voting Classifier or a Pipeline without writing a single line of wrapper code.
* **Batch Processing:** "From scratch" tutorials usually crash the moment you feed them a dataset larger than your GPU memory. I prioritized system stability by adding automatic batching and explicit memory freeing in `predict_proba`. You can run inference on datasets of infinite size (limited only by your hard drive), making this viable for actual production deployment on edge devices.
* **Robust Convergence Mechanisms:** I didn't want a toy model that diverges on real data; so i created the essential safety rails: Early Stopping, Patience, Learning Rate Decay, and Gradient Clipping. These aren't just "nice to haves"; they are the engineering controls that ensure the model actually converges on non-trivial tasks.

## References & Inspiration

My journey was guided by these core resources, which I studied to understand the "why" behind the code:

1.  **Sebastian Raschka's Lecture Notes:** [Specifically L8 on Logistic Regression & Softmax.](https://sebastianraschka.com/pdf/lecture-notes/stat453ss21/L08_logistic__slides.pdf)
2.  **Xavier Glorot & Yoshua Bengio (2010):** ["Understanding the difficulty of training deep feedforward neural networks."](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
3.  **3Blue1Brown:** [Neural Networks intuitive illustration.](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&si=TSHHNefSiJ-UNxCD)
4.  **Scikit-learn's API Reference:** For [`ClassifierMixin`](https://scikit-learn.org/stable/modules/generated/sklearn.base.ClassifierMixin.html) And [`BaseEstimator`](https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html) Classes.
5.  **CuPy's User Guide:** [Basics of cupy.ndarray.](https://docs.cupy.dev/en/stable/user_guide/basic.html)
