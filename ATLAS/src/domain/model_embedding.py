from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling
import torch
from transformers import AutoTokenizer, AutoModel
#from transformers import GPT2Model, GPT2Tokenizer
from transformers import BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer
'''
class ModelEmbedding:
    def __init__(self, embedding_model_name):
        # Forziamo CPU per stabilità con architettura Blackwell (RTX 5060)
        self.device = "cuda"
        self.model = SentenceTransformer(embedding_model_name, device=self.device)
        print(f"Modello caricato correttamente su: {self.device}")

    def encode(self, sentence):
        # Calcolo su CPU
        embedding = self.model.encode(
            sentence,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=1
        )
        # Restituiamo il dizionario compatibile con River
        return embedding#{f"f{i}": float(v) for i, v in enumerate(embedding)}
'''

#GPT2
class ModelEmbedding:
    def __init__(self, embedding_model_name):
        #self.model = SentenceTransformer(embedding_model_name)
        self.model = AutoModel.from_pretrained('bert-base-uncased', local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', truncation_side='left', local_files_only=True)
        #self.tokenizer.pad_token = self.tokenizer.eos_token #GPT2

        self.device = 'cuda'
        self.model.to('cuda')
        self.model.eval()

    def encode(self, sentence):
        #embedding = self.model.encode(sentence)  # np.ndarray
        
        inputs = self.tokenizer(sentence, return_tensors='pt', truncation=True,  padding=True,
        max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward (no gradients)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Estrai last hidden state dell'ultimo token
        # Questo contiene context di tutta la sequenza
        #last_hidden = outputs.last_hidden_state #gpt2
        #embedding = last_hidden[0, -1, :].cpu().numpy() #gpt2
        embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()  # (1, 768)
        
        #feature_dict = { i: v for i, v in enumerate(embedding)}
        return embedding#feature_dict

'''
import fasttext
import numpy as np
class ModelEmbedding:
    def __init__(self, embedding_model_name):
        # Carica modello FastText pretrained
        # embedding_model_name dovrebbe essere il path al file .bin
        # Es: 'cc.it.300.bin' per italiano
        self.model = fasttext.load_model('cc.en.300.bin')
    
    def encode(self, sentence):
        # FastText genera embedding direttamente
        embedding = self.model.get_sentence_vector(sentence)
        
        # Converti in dictionary con chiavi f0, f1, ...
        feature_dict = {f"f{i}": float(v) for i, v in enumerate(embedding)}
        return feature_dict
    
'''