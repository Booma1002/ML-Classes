import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, roc_curve, auc, hamming_loss, precision_recall_curve, \
    average_precision_score
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef, hamming_loss



class Visualizer:
    @staticmethod
    def plot_grid(classifiers, plot_func, X_test, y_test, shape=(1, 1), figsize=(15, 10), title=None, save_path=None):
        fig, axes = plt.subplots(shape[0], shape[1], figsize=figsize)
        if title: fig.suptitle(title, fontsize=20, fontweight='bold')
        axes_flat = axes.flatten() if shape[0] * shape[1] > 1 else [axes]

        for (name, clf), ax in zip(classifiers.items(), axes_flat):
            if plot_func == plot_training_history:
                plot_func(clf, ax=ax, title=f"{name} History")
            elif plot_func == conf:
                y_pred = clf.predict(X_test)
                plot_func(clf, y_test, y_pred, ax=ax)
                ax.set_title(f"{name} Confusion Matrix", fontsize=16)
            elif plot_func == ROC or plot_func == plot_precision_recall:
                plot_func(clf, y_test, X_test, ax=ax)
                ax.set_title(f"{name} {plot_func.__name__}", fontsize=16)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        plt.show()


def plot_training_history(clf, scope=50, title="Model Training History", ax_loss=None):
    if ax_loss is None:
        fig, ax_loss = plt.subplots(figsize=(12, 7))
        should_show = True
    else:
        should_show = False

    def get_series(attribute_name):
        if not hasattr(clf, attribute_name): return None
        raw_data = getattr(clf, attribute_name)
        if not raw_data or len(raw_data) == 0: return None
        clean_data = [x.item() if hasattr(x, 'item') else x for x in raw_data]
        y_full = np.array(clean_data)
        idx = np.arange(0, len(y_full), scope)
        if idx[-1] != len(y_full) - 1: idx = np.append(idx, len(y_full) - 1)
        return idx + 1, y_full[idx]

    ax_score = ax_loss.twinx()

    if title:
        ax_loss.set_title(title, fontsize=16, fontweight='bold')

    lines = []
    res = get_series('losses_') or get_series('loss_curve_')
    if res:
        x, y = res
        ln, = ax_loss.plot(x, y, color='crimson', lw=4, linestyle='-', alpha=0.8, label='Train Loss')
        lines.append(ln)

    res = get_series('val_losses_')
    if res:
        x, y = res
        if hasattr(clf, 'val_jump'): x = x * clf.val_jump
        min_val = np.min(y)
        ln, = ax_loss.plot(x, y, color='red', lw=4, linestyle='--', marker='o', markersize=7,
                           label=f'Val Loss (Min: {min_val:.4f})')
        lines.append(ln)

    res = get_series('scores_')
    if res:
        x, y = res
        ln, = ax_score.plot(x, y, color='dodgerblue', linestyle='-', linewidth=4, alpha=0.6, label='Train Score')
        lines.append(ln)

    res = get_series('val_scores_') or get_series('validation_scores_')
    if res:
        x, y = res
        if hasattr(clf, 'val_jump'): x = x * clf.val_jump
        max_val = np.max(y)
        ln, = ax_score.plot(x, y, color='navy', linewidth=4, linestyle='--', marker='s', markersize=7,
                            label=f'Val Score (Max: {max_val:.4f})')
        lines.append(ln)

    if hasattr(clf, "best_epoch_") and clf.best_epoch_:
        vline = ax_loss.axvline(clf.best_epoch_, color='green', linestyle='-.', linewidth=2,
                                label=f'Best Epoch: {clf.best_epoch_}')
        lines.append(vline)

    ax_loss.set_xlabel("Epoch", fontsize=14)
    ax_loss.tick_params(axis='x', labelsize=14)
    ax_loss.set_ylabel("Loss (Cross Entropy)", fontsize=16, color='crimson')
    ax_loss.tick_params(axis='y', labelcolor='crimson', labelsize=14)
    ax_loss.grid(True, linestyle=':', alpha=0.6)

    ax_score.set_ylabel("Score", fontsize=16, color='navy')
    ax_score.tick_params(axis='y', labelcolor='navy', labelsize=14)
    ax_score.set_ylim(0, 1.05)

    labels = [l.get_label() for l in lines]
    ax_loss.legend(lines, labels, loc="right", frameon=True, shadow=True)

    if should_show:
        plt.tight_layout()
        plt.show()

def conf(clf, y_test, y_pred, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        should_show = True
    else:
        should_show = False

    if hasattr(clf, 'classes_'):
        classes = clf.classes_
    else:
        classes = np.unique(y_test)

    cm = confusion_matrix(y_test, y_pred)
    y_test_bin = label_binarize(y_test, classes=classes)
    y_pred_bin = label_binarize(y_pred, classes=classes)

    specs = []
    n_classes = y_test_bin.shape[1]
    if n_classes == 1 and len(classes) == 2:
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specs.append(tn / (tn + fp))
    else:
        for i in range(n_classes):
            tn, fp, fn, tp = confusion_matrix(y_test_bin[:, i], y_pred_bin[:, i]).ravel()
            specs.append(tn / (tn + fp))

    spec = np.mean(specs)
    hamming = hamming_loss(y_test, y_pred)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Spectral', cbar=False, ax=ax)

    ax.set_title(f'Confusion Matrix\nMacro-Avg Specificity: [{spec:.4f}]  -  Hamming-loss: [{hamming:.4f}]',
                 fontsize=18)
    ax.set_ylabel('True Label', fontsize=14, color='crimson')
    ax.set_xlabel('Predicted Label', fontsize=14, color='navy')
    ax.tick_params(axis='y', labelcolor='crimson', labelsize=14)
    ax.tick_params(axis='x', labelcolor='navy', labelsize=14)

    if should_show:
        plt.show()


def ROC(clf, y_test, X_test_proc, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        should_show = True
    else:
        should_show = False

    y_test_bin = label_binarize(y_test, classes=clf.classes_)
    y_score = clf.predict_proba(X_test_proc)
    n_classes = y_test_bin.shape[1]

    if n_classes == 1:
        n_classes = 2
        y_test_bin = np.hstack((1 - y_test_bin, y_test_bin))

    colors = plt.get_cmap('coolwarm', n_classes)

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors(i), lw=1, label=f'Digit {i} (AUC = {roc_auc:.6f})')

    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.05, 1.0)
    ax.plot([0, 1], [0, 1], 'g--', lw=2)
    ax.set_title('ROC Curve', fontsize=20)
    ax.set_xlabel('False Positive Rate', fontsize=18, color='crimson')
    ax.set_ylabel('True Positive Rate', fontsize=18, color='navy')
    ax.tick_params(axis='x', labelcolor='crimson', labelsize=14)
    ax.tick_params(axis='y', labelcolor='navy', labelsize=14)
    ax.legend(loc="lower right", frameon=True, shadow=True, fontsize=8)
    ax.grid(alpha=0.4, axis="both", color='k', lw=2)

    if should_show:
        plt.show()


def plot_precision_recall(clf, y_test, X_test_proc, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        should_show = True
    else:
        should_show = False

    y_test_bin = label_binarize(y_test, classes=clf.classes_)
    y_score = clf.predict_proba(X_test_proc)
    n_classes = y_test_bin.shape[1]

    if n_classes == 1:
        n_classes = 2
        y_test_bin = np.hstack((1 - y_test_bin, y_test_bin))

    colors = plt.get_cmap('tab10', n_classes)

    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        prec = average_precision_score(y_test_bin[:, i], y_score[:, i])

        ax.plot(recall, precision, color=colors(i), lw=2,
                label=f'Digit {i} (AP = {prec:.4f})')

    ax.set_xlabel('Recall', fontsize=24)
    ax.set_ylabel('Precision', fontsize=24)
    ax.set_title('Precision-Recall Curve', fontsize=18)
    ax.tick_params(labelsize=12)
    ax.legend(loc="best", fontsize=12, frameon=True, shadow=True)
    ax.grid(alpha=0.8)

    if should_show:
        plt.tight_layout()
        plt.show()

def plot_benchmark_metrics(classifiers, X_test, y_test, ax=None, save_path=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))
        should_show = True
    else:
        fig= ax.get_figure()
        should_show = False

    metrics_names = ['Precision', 'Recall', 'F1 Score', 'MCC', 'Hamming Loss', 'Specificity']
    results = {name: [] for name in classifiers.keys()}
    for name, clf in classifiers.items():
        y_pred = clf.predict(X_test)


        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        hamming = hamming_loss(y_test, y_pred)
        classes = np.unique(y_test)
        specs = []
        y_test_bin = label_binarize(y_test, classes=classes)
        y_pred_bin = label_binarize(y_pred, classes=classes)

        for i in range(len(classes)):
            if y_test_bin.shape[1] == 1:
                t_true, t_pred = y_test_bin.ravel(), y_pred_bin.ravel()
            else:
                t_true, t_pred = y_test_bin[:, i], y_pred_bin[:, i]
            tn, fp, fn, tp = confusion_matrix(t_true, t_pred).ravel()
            specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        spec = np.mean(specs)
        results[name] = [prec, rec, f1, mcc, hamming, spec]

    x = np.arange(len(metrics_names))
    width = 0.35
    multiplier = 0

    for attribute, measurement in results.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3, fmt='%.3f', fontsize=10, fontweight='bold')
        multiplier += 1

    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('Benchmark Metrics Comparison', fontsize=18, fontweight='bold')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metrics_names, fontsize=12)
    ax.legend(loc='lower left', fontsize=12, frameon=True, shadow=True)
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    if should_show:
        plt.tight_layout()
        plt.show()