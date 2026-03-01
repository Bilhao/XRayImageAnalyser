import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve


def evaluate(model, test_gen, class_names, output_dir="./output"):
    os.makedirs(output_dir, exist_ok=True)
    
    y_pred = model.predict(test_gen, verbose=1)
    
    y_true = []
    for i in range(len(test_gen)):
        _, labels = test_gen[i]
        y_true.append(labels)
    y_true = np.concatenate(y_true, axis=0)
    
    y_true = y_true[:len(y_pred)]
    
    print(f"\n{'Classes':<25} {'AUC-ROC':>8}")
    print("-" * 35)
    
    auc_scores = {}
    for i, name in enumerate(class_names):
        try:
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
            auc_scores[name] = auc
            print(f"{name:<25} {auc:>8.4f}")
        except ValueError:
            auc_scores[name] = float('nan')
            print(f"{name:<25} {'N/A':>8}")
    
    valid_aucs = [v for v in auc_scores.values() if not np.isnan(v)]
    mean_auc = np.mean(valid_aucs) if valid_aucs else 0.0
    print("-" * 35)
    print(f"{'Média dos AUC':<25} {mean_auc:>8.4f}")
    
    plot_rocs(y_true, y_pred, class_names, output_dir)
    
    return auc_scores


def plot_rocs(y_true, y_pred, class_names, output_dir="./output"):
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, name in enumerate(class_names):
        ax = axes[i]
        try:
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
            ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        except ValueError:
            ax.text(0.3, 0.5, "N/A", fontsize=14)
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel('False Positive Rate', fontsize=8)
        ax.set_ylabel('True Positive Rate', fontsize=8)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)
    
    for j in range(len(class_names), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "roc_curves.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
