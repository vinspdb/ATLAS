from datetime import datetime

from src.domain.event import Event
import csv

"""
Classe responsabile di leggere il file CSV in modalità streaming.
Carica l'header, legge una riga alla volta e crea Event.
Restituisce None quando il file è terminato.
"""
class StreamDataLoader:
    def __init__(self, path):
        self.file = open(path, mode="r", encoding="utf-8", newline="")
        self.source = csv.DictReader(self.file, delimiter=";")

    def read_event(self):
        try:
            row = next(self.source)
        except StopIteration:
            return None

        case_id = row["Case ID"]
        activity = row["Activity"]
        resource = row['Resource']

        # parsing timestamp QUI
        timestamp_str = row["Complete Timestamp"]
        timestamp = datetime.fromisoformat(
            timestamp_str.replace("Z", "+00:00")
        )

        attributes = {k: v for k, v in row.items()
                      if k not in ("Case ID", "Activity", "Complete Timestamp", 'Resource')}

        return case_id, Event(activity, timestamp, resource, attributes)

    def close_stream(self):
        if self.file:
            self.file.close()
            self.file = None
            self.source = None