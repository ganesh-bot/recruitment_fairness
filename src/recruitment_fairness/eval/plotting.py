import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix

def plot_delta_tpr(df, save_path=None):
    """
    df: DataFrame with columns ['model','delta_tpr']
    """
    ax = df.plot.bar(x='model', y='delta_tpr', legend=False)
    ax.set_ylabel("ΔTPR")
    ax.set_xlabel("Model")
    ax.set_title("Fairness gap by model")
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    return ax

def plot_roc_curves(curve_data, save_path=None):
    """
    curve_data: dict of {model_name: (fpr, tpr)}
    """
    plt.figure()
    for name, (fpr, tpr) in curve_data.items():
        plt.plot(fpr, tpr, label=name)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

def plot_delta_tpr(summary_df, save_path=None):
    ax = summary_df.plot.bar(x="model", y="delta_tpr", legend=False)
    ax.set_ylabel("ΔTPR")
    ax.set_xlabel("Model")
    ax.set_title("Fairness Gap by Model")
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    return ax

def fairness_report_by_group(
    df: pd.DataFrame,
    true_col: str,
    pred_col: str,
    group_cols: list[str],
    out_dir: str,
):
    """
    For each group in group_cols, computes TPR per group and ΔTPR,
    writes a JSON summary and a bar-chart PNG for each grouping.
    """
    import os, json
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    report = []

    for grp in group_cols:
        vals = []
        for g in df[grp].dropna().unique():
            mask   = df[grp] == g
            y_true = df.loc[mask, true_col]
            y_pred = df.loc[mask, pred_col]
            tp = ((y_true==1)&(y_pred==1)).sum()
            fn = ((y_true==1)&(y_pred==0)).sum()
            tpr = tp/(tp+fn) if (tp+fn)>0 else np.nan
            vals.append({"group": str(g), "tpr": float(tpr)})

        delta = float(np.nanmax([v["tpr"] for v in vals]) 
                      - np.nanmin([v["tpr"] for v in vals]))
        report.append({"by": grp, "delta_tpr": delta, "groups": vals})

        # plot
        chart = pd.DataFrame(vals).set_index("group")
        ax = chart.plot.bar(
            y="tpr", legend=False, title=f"TPR by {grp} (ΔTPR={delta:.3f})"
        )
        ax.set_ylabel("TPR")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"tpr_by_{grp}.png"))
        plt.close()

    # write master JSON
    with open(os.path.join(out_dir, "fairness_by_group.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"✅ Wrote fairness report to {out_dir}/fairness_by_group.json")
