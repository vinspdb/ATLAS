
"""
Rappresenta un singolo evento del log.
Contiene case_id, activity, timestamp, label e attributi aggiuntivi.
Determina se un evento è finale controllando il suffisso _end del case_id.
"""
class Event:
    def __init__(self, activity, timestamp, resource, attributes=None):
        self.activity = activity
        self.timestamp = timestamp
        self.attributes = attributes
        self.resource = resource

    def get_activity(self):
        return self.activity

    def get_timestamp(self):
        return self.timestamp

    def __get_time_from_start_ms(self, timestamp_start):
        delta = self.timestamp - timestamp_start
        time_from_start_ms = int(delta.total_seconds())
        return time_from_start_ms

    def __get_attributes(self, key):
        return self.attributes[key]

    def is_final_event(self):
        return self.activity == "END"

    def to_string(self, timestamp_start):
        if self.is_final_event():
            return ""

        parts = [
            f"{self.activity}",
            #f"{self.__get_time_from_start_ms(timestamp_start)}",
            #f"{self.resource}"
            #f"{self.__get_time_from_start_ms(timestamp_start)} days since case start.",
            #f"{self.resource}"
        ]

        #context = []
        #if self.attributes:
        #    for k, v in self.attributes.items():
        #        parts.append(f"{k}={v}")

        return " ".join(parts)




