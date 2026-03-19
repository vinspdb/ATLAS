import yaml

"""
Classe responsabile di leggere il file di configurazione config.yaml e fornire accesso 
ai dati necessari (csv_path, embedding_model_name, classifier_name)
"""
class ConfigYaml:
    def __init__(self, yaml_path):
        self.csv_path = None
        self.embedding_model_name = None
        self.classifier_type = None
        self.classifier_params = {}
        self.evaluation_params = {}

        with open(yaml_path, "r") as f:
            properties = yaml.safe_load(f)

        self.csv_path = properties["dataset"]["path"]
        self.embedding_model_name = properties["embedded_model"]["model"]

        classifier_cfg = properties["river"]["classifier"]
        self.classifier_type = classifier_cfg["type"]

        # tutti gli altri campi sono parametri del classifier
        self.classifier_params = {
            k: v for k, v in classifier_cfg.items() if k != "type"
        }

        self.evaluation_params = properties.get("evaluation", {})

        f.close()

    def get_csv_path(self):
        return self.csv_path

    def get_embedding_model_name(self):
        return self.embedding_model_name

    def get_classifier_type(self):
        return self.classifier_type

    def get_classifier_params(self):
        return self.classifier_params

    def get_evaluation_params(self):
        return self.evaluation_params


