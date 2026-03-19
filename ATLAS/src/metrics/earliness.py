import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import recall_score, auc

def skip_first_cases(df: pd.DataFrame, skip_ratio: float) -> pd.DataFrame:
    case_ids = df["case_id"].dropna().unique()
    cut = int(len(case_ids) * skip_ratio)
    keep = set(case_ids[cut:])
    return df[df["case_id"].isin(keep)].copy()

def compute_earliness_auc(df: pd.DataFrame, pos_label: str, skip_ratio: float):
    df = skip_first_cases(df, skip_ratio)

    df["y_true"] = (df["true_label"] == pos_label).astype(int)
    df["y_pred"] = (df["predicted_label"] == pos_label).astype(int)

    xs, ys = [], []

    prefix_lengths = sorted(df["prefix_len"].unique())
    for p in prefix_lengths:
        sub = df[df["prefix_len"] == p]

        if sub["y_true"].sum() == 0:
            r = 0.0
        else:
            r = recall_score(sub["y_true"], sub["y_pred"], zero_division=0)

        xs.append(p)
        ys.append(r)

    auc_value = auc(xs, ys) if len(xs) > 1 else 0.0
    return xs, ys, auc_value

def plot_earliness_curve(xs, ys, auc_value, title, pos_label):
    plt.figure(figsize=(18, 6))
    plt.plot(xs, ys, marker="o", markersize=3, linewidth=1)

    plt.xlabel("Prefix length")
    plt.ylabel(f"Recall ({pos_label})")
    plt.title(f"{title} | Earliness AUC = {auc_value:.3f}")
    plt.grid(True)
    plt.ylim(0, 1)

    step = max(1, len(xs) // 15)
    plt.xticks(xs[::step])

    plt.tight_layout()
    #plt.savefig(f"{title}_earliness.png", dpi=300, bbox_inches="tight")
    plt.show()
