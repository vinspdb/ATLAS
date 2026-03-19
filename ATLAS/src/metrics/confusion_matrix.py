import pandas as pd
from sklearn.metrics import  classification_report


def plot_multiclass_confusion_matrix(df: pd.DataFrame, title: str, skip_ratio: float):
    unique_cases = df#.unique()
    num_to_skip = skip_ratio#int(len(unique_cases) * skip_ratio)
    cases_to_include = unique_cases#[num_to_skip:]
    print('skip-ratio',num_to_skip)
    df_cm = cases_to_include
    if df_cm.empty:
        print("Dataset vuoto dopo lo skip_ratio. Impossibile generare la matrice.")
        return

    y_true = df_cm["true_activity"]
    y_pred = df_cm["predicted_activity"]

    # 4. Calcolo metriche
    cr = classification_report(y_true, y_pred, digits=3)
    print(cr)

    