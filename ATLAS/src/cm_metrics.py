import argparse
import pandas as pd
from src.metrics.confusion_matrix import plot_multiclass_confusion_matrix

def compute_metrics(config_path):
    event_log = pd.read_csv(f"src/dataset/csv/{args.log}.csv", sep=';')
    skip_ratio = (len(event_log)//100)*10
    print(skip_ratio)

    print("Event Log--->",args.log)

    paths = [
        f"src/output_new/NN_{args.log}_evaluation.csv",
    ]

    for path in paths:
        df = pd.read_csv(path)
        plot_multiclass_confusion_matrix(df, "NN", skip_ratio)

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="../config.yaml",
        help="Path al file di configurazione"
    )
    parser.add_argument(
        "--log",
        help="Event log"
    )
    
    args = parser.parse_args()

      
    compute_metrics(args.config)
