import argparse
import csv
import time
import os
import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from river import drift

from src.domain.active_case_manager import ActiveCaseManager
from src.domain.config_yaml import ConfigYaml
from src.domain.model_embedding import ModelEmbedding
from src.domain.stream_classifier import StreamClassifier
from src.domain.stream_data_loader import StreamDataLoader

SEED = 42


def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed fissato a {seed}")


def encode_prefix(trace_string: str, model_embedding: ModelEmbedding) -> dict:
    """Encode the last 4 lines of a trace string into a feature dict."""
    lines = trace_string.split('\n')[-4:]
    embedding = model_embedding.encode("->".join(lines))
    return {i: v for i, v in enumerate(np.array(list(embedding)))}


def handle_drift(
    event_count: int,
    drift_count: int,
    example_dict: dict,
    stream_classifier: StreamClassifier,
    drift_writer,
    drift_file,
) -> tuple[int, dict]:
    """
    Execute retraining on the ADWIN window and log the drift event.
    Returns (updated drift_count, updated example_dict).
    """
    drift_count += 1
    window_width = int(stream_classifier.error_drift.width)

    print(f"Drift #{drift_count} rilevato a evento {event_count}")
    print(f"   Finestra ADWIN width: {window_width}")

    first_event = event_count - window_width + 1
    X_batch, y_batch, window_events = [], [], []

    for evt in range(max(1, first_event), event_count + 1):
        if evt in example_dict:
            X_batch.append(example_dict[evt]['embedding'])
            y_batch.append(example_dict[evt]['label'])
            window_events.append(evt)

    if window_events:
        print(f"   Retraining su eventi: {window_events[0]} a {window_events[-1]}")

    if X_batch:
        X_batch_df = pd.DataFrame(X_batch)
        stream_classifier.classifier[-1].module.reset_head()
        stream_classifier.learn_many(X_batch_df, y_batch)
        print(f"Retraining completato con {len(X_batch)} esempi")

    drift_writer.writerow([
        drift_count,
        event_count,
        window_width,
        time.strftime('%Y-%m-%d %H:%M:%S'),
    ])
    drift_file.flush()

    # Drop examples outside the current ADWIN window
    for e in [e for e in example_dict if e < first_event]:
        del example_dict[e]

    return drift_count, example_dict


def run(args):
    config_yaml = ConfigYaml(args.config)
    csv_path = config_yaml.get_csv_path()

    loader = StreamDataLoader(csv_path)
    manager = ActiveCaseManager()
    model_embedding = ModelEmbedding(config_yaml.get_embedding_model_name())

    event_log = pd.read_csv(csv_path, sep=';')
    skip_ratio = (len(event_log) // 100) * 10
    total_events = len(event_log)

    stream_classifier = StreamClassifier(
        config_yaml.get_classifier_type(),
        config_yaml.get_classifier_params(),
    )
    

    out_dir = "output_new"
    os.makedirs(out_dir, exist_ok=True)

    eval_path  = os.path.join(out_dir, f"NN_{args.log}_evaluation.csv")
    drift_path = os.path.join(out_dir, f"drift_log_{args.log}.csv")

    with (
        open(eval_path,  mode="w", newline="", encoding="utf-8") as eval_file,
        open(drift_path, mode="w", newline="", encoding="utf-8") as drift_file,
    ):
        eval_writer  = csv.writer(eval_file)
        drift_writer = csv.writer(drift_file)

        eval_writer.writerow(["case_id", "trace_len", "predicted_activity", "true_activity"])
        drift_writer.writerow(["drift_number", "event_count", "window_width", "timestamp"])

        error_drift = drift.ADWIN()
        stream_classifier.error_drift = error_drift

        example_dict: dict = {}
        event_count = 0
        drift_count = 0

        pbar = tqdm(total=total_events, desc="Elaborazione eventi", unit="evt")

        while True:
            result = loader.read_event()
            if result is None:
                break

            event_count += 1
            after_cut = event_count > skip_ratio

            case_id, event = result
            manager.add_event(case_id=case_id, event=event)

            if not after_cut:
                manager.set_prediction(case_id, None)

            completed  = manager.is_trace_completed(case_id)
            trace      = manager.get_trace(case_id)
            trace_len  = trace.get_length()

            if completed:
                # ---- log prediction vs END ----
                if after_cut:
                    pred = manager.get_activity_prediction(case_id)
                    if pred is not None:
                        eval_writer.writerow([case_id, trace_len, pred, "END"])

                y_pred = manager.get_activity_prediction(case_id)
                y_true = event.get_activity()
                error  = int(y_pred != y_true)

                prefix_trace, last_activity = trace.get_prefix_trace()
                embedding = encode_prefix(prefix_trace.to_string(), model_embedding)

                example_dict[event_count] = {'embedding': embedding, 'label': last_activity}
                error_drift.update(error)

                if error_drift.drift_detected:
                    drift_count, example_dict = handle_drift(
                        event_count, drift_count, example_dict,
                        stream_classifier, drift_writer, drift_file,
                    )
                else:
                    stream_classifier.learn_one(embedding, last_activity)

                manager.close_trace(case_id)

            else:
                # ---- active trace ----
                if trace_len > 1:
                    if after_cut:
                        pred = manager.get_activity_prediction(case_id)
                        if pred is not None:
                            eval_writer.writerow(
                                [case_id, trace_len, pred, event.get_activity()]
                            )

                    y_pred = manager.get_activity_prediction(case_id)
                    y_true = event.get_activity()
                    error  = int(y_pred != y_true)

                    prefix_trace, last_activity = trace.get_prefix_trace()
                    embedding = encode_prefix(prefix_trace.to_string(), model_embedding)

                    example_dict[event_count] = {'embedding': embedding, 'label': last_activity}
                    error_drift.update(error)

                    if error_drift.drift_detected:
                        drift_count, example_dict = handle_drift(
                            event_count, drift_count, example_dict,
                            stream_classifier, drift_writer, drift_file,
                        )
                    else:
                        stream_classifier.learn_one(embedding, last_activity)

                if after_cut:
                    trace_embedding = encode_prefix(trace.to_string(), model_embedding)
                    predicted = stream_classifier.predict_one(trace_embedding)
                    manager.set_prediction(case_id, predicted)

            pbar.update(1)

        pbar.close()


if __name__ == "__main__":
    set_seed(SEED)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config.yaml", help="Path al file di configurazione")
    parser.add_argument("--log",    required=True, help="Nome dell'event log")

    args = parser.parse_args()

    start = time.time()
    run(args)
    elapsed = time.time() - start

    with open(f"{args.log}_time_atlas.txt", "w") as tf:
        tf.write(str(elapsed))

    print(f"Tempo totale: {elapsed:.2f}s")