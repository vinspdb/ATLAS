
from src.domain.event import Event

"""
Rappresenta la sequenza ordinata di eventi di un case.
Mantiene la lista degli eventi e permette di aggiungerne di nuovi.
"""
class Trace:
    def __init__(self, case_id):
        self.case_id = case_id
        self.events = []

    def add_event(self, event):
        self.events.append(event)

    def is_trace_completed(self):
        if not self.events:
            return False
        return self.events[-1].is_final_event()

    def get_length(self):
        return len(self.events)

    def get_prefix_trace(self):
        """
        Restituisce un nuovo oggetto Trace contenente tutti gli eventi
        tranne l'ultimo, e l'attività dell'ultimo evento.
        Se c'è solo un evento, il prefisso sarà una Trace vuota.
        """
        # Se non ci sono eventi, gestiamo il caso limite (opzionale)
        if not self.events:
            return None, None

        # 1. Creiamo la nuova Trace (il prefisso)
        prefix_trace = Trace(self.case_id)

        # 2. Prendiamo tutti gli eventi tranne l'ultimo: self.events[:-1]
        for e in self.events[:-1]:
            # Copiamo l'evento (o passiamo il riferimento se non serve copiarlo)
            new_e = Event(
                activity=e.activity,
                timestamp=e.timestamp,
                resource = e.resource,
                attributes=e.attributes

            )
            prefix_trace.add_event(new_e)

        # 3. Prendiamo l'activity dell'ultimo evento
        # TODO: Siamo sicuriiiiiii????
        last_activity = self.events[-1].activity

        # Ritorna la singola Trace prefisso e la stringa dell'attività
        return prefix_trace, last_activity

    def to_string(self):
        timestamp_start = self.events[0].get_timestamp()
        lines = []
        for e in self.events:
            event_str = e.to_string(timestamp_start)
            if event_str:
                lines.append(f"{event_str}")
        return "\n".join(lines)
