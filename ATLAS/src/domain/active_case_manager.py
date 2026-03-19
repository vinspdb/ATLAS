from src.domain.trace import Trace
from collections import Counter
import math
from src.domain.model_embedding import ModelEmbedding
from src.domain.config_yaml import ConfigYaml
import numpy as np
config_yaml = ConfigYaml("/home/vincenzo/Documenti/outcome_streaming/Tesi-next_activity/config.yaml")

"""
Classe che gestisce i casi attivi durante lo streaming e relative label.
Mantiene un dizionario case_id → [Trace, List<String>]
Permette di chiudere e rimuovere la trace.
"""

model_embedding = ModelEmbedding(config_yaml.get_embedding_model_name())

class ActiveCaseManager:
    def __init__(self):
        self.active_cases = {}  # case_id -> [Trace, str]

    def get_trace(self, case_id):
        return self.active_cases[case_id][0]

    def get_activity_prediction(self, case_id):
        return self.active_cases[case_id][1]

    def add_event(self, case_id, event):
        if case_id not in self.active_cases:
            # Inizializzo con una stringa vuota o un valore di default (es. None)
            self.active_cases[case_id] = [Trace(case_id), ""]
        self.active_cases[case_id][0].add_event(event)

    def set_prediction(self, case_id, predicted_activity):
        # Sovrascrive la stringa esistente invece di fare .append()
        self.active_cases[case_id][1] = predicted_activity

    def is_trace_completed(self, case_id):
        return self.active_cases[case_id][0].is_trace_completed()

    def close_trace(self, case_id):
        del self.active_cases[case_id]
    
    def get_all_case(self):
        if not self.active_cases:
            return [0,0,0]
        trace_lengths = []
        unique_traces = set()
        all_activities = set()    
        #all_activities = []
        for trace, _ in self.active_cases.values():
                activities = [a for a in trace.to_string().split('\n') if a.strip()]#[-4:]
        
                trace_lengths.append(len(activities))

                unique_traces.add(" ".join(activities))

                all_activities.add('->'.join(activities))
        #active_embeddings = model_embedding.encode(all_activities)
        
        #if len(all_activities)>0:
        #    global_embedding = np.sum(active_embeddings, axis=0)  # shape (embedding_dim,)
        #else:
        #    global_embedding = np.zeros(model_embedding.embedding_dim)

        #return global_embedding#dict(enumerate(global_embedding))
        
        num_active = len(self.active_cases)
        num_unique = len(unique_traces)

        variability = num_unique / num_active
        num_distinct_activities = len(all_activities)

        #print(num_active, variability, num_distinct_activities)
        return [num_active, variability, num_distinct_activities]#{0:num_active, 1:variability, 2:num_distinct_activities}
        
        '''
        (
            f"Active cases: {num_active} | "
            f"Avg trace length: {avg_length:.2f} | "
            #f"Unique traces: {num_unique} | "
            #f"Variability: {variability:.2f} | "
            f"Confidence: {status}"
        )'''