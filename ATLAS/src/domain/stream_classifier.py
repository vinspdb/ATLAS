from river import forest, tree, ensemble, drift, metrics
from river import metrics, datasets, preprocessing, compose
from deep_river.classification import Classifier, RollingClassifier
from deep_river.classification.zoo import LSTMClassifier
from torch import nn
from torch import optim
from torch import manual_seed

_ = manual_seed(42)
n_features = 768

from torch import nn

import numpy as np
import torch
from torch import nn
from transformers import AutoModel
from river import base, drift


class BertClassifierModule(nn.Module):
    def __init__(self, model_name, max_length=512, num_classes=2):
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        self.hidden = self.bert.config.hidden_size

        self.classifier = nn.Linear(self.hidden, num_classes)

    def forward(self, x):
        # x shape: (batch, 2 * max_length)
        input_ids = x[:, :self.max_length].long()
        attention_mask = x[:, self.max_length:].long()

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls)

class MyConv1DDeepModule(nn.Module):
    def __init__(self, n_features, hidden_channels=[128, 256], num_classes=2,
                 use_mlp=False, mlp_hidden=128, kernel_size=3):
        """
        n_features     : numero di feature in input
        hidden_channels: lista di canali Conv1D per ogni layer
        num_classes    : numero di classi
        use_mlp        : se usare MLP tra Conv1D e FC
        mlp_hidden     : hidden size MLP
        kernel_size    : dimensione kernel Conv1D
        """
        super().__init__()

        layers = []
        in_channels = n_features
        for out_channels in hidden_channels:
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2
                )
            )
            layers.append(nn.ReLU())
            # GroupNorm con 1 gruppo funziona anche con seq_len=1 e batch=1
            layers.append(nn.GroupNorm(num_groups=1, num_channels=out_channels))
            in_channels = out_channels

        self.conv = nn.Sequential(*layers)

        # MLP opzionale
        self.use_mlp = use_mlp
        if use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(hidden_channels[-1], mlp_hidden),
                nn.ReLU(),
                nn.Linear(mlp_hidden, hidden_channels[-1]),
                nn.ReLU()
            )

        # Testa di classificazione
        self.fc = nn.Linear(hidden_channels[-1], num_classes)

    def forward(self, x):
        """
        x: tensor [batch, n_features] oppure [batch, seq_len, n_features]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, seq_len=1, n_features]

        # Conv1D vuole [batch, channels, seq_len]
        x = x.transpose(1, 2)  # [batch, n_features, seq_len]
        x = self.conv(x)       # [batch, hidden_channels[-1], seq_len]

        # aggregazione lungo seq_len per ottenere vettore fisso
        h_last = x.mean(dim=2)  # [batch, hidden_channels[-1]]

        # passaggio MLP se presente
        if self.use_mlp:
            h_last = self.mlp(h_last)

        out = self.fc(h_last)  # [batch, num_classes]
        return out

    def reset_head(self):
        """
        Reset selettivo di MLP + FC senza toccare Conv1D
        """
        with torch.no_grad():
            if self.use_mlp:
                for layer in self.mlp:
                    if hasattr(layer, "reset_parameters"):
                        layer.reset_parameters()
            self.fc.reset_parameters()

class MyRNNModule(nn.Module):
    def __init__(self, n_features, hidden_size=256, num_classes=2,
                 num_layers=1, use_mlp=True, mlp_hidden=128):
        """
        n_features : int      → numero di feature in input
        hidden_size: int      → hidden size GRU
        num_classes: int      → numero di classi
        num_layers: int       → numero di layer GRU
        use_mlp   : bool      → se usare MLP tra GRU e FC
        mlp_hidden: int       → hidden size MLP
        """
        super().__init__()

        # GRU multilayer
        self.rnn = nn.RNN(
            input_size=n_features,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=num_layers
        )

        # MLP opzionale prima della testa
        self.use_mlp = use_mlp
        if use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size, mlp_hidden),
                nn.ReLU(),
                nn.Linear(mlp_hidden, hidden_size),
                nn.ReLU()
            )

        # Testa di classificazione
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        x: tensor (batch, n_features)
        """
        # aggiungo dimensione seq_len=1
        x = x.unsqueeze(1)  # (batch, seq_len=1, n_features)
        _, h_n = self.rnn(x)  # h_n: (num_layers, batch, hidden_size)

        # prendo solo l'output dell'ultimo layer
        h_last = h_n[-1]  # (batch, hidden_size)

        # passaggio MLP se presente
        if self.use_mlp:
            h_last = self.mlp(h_last)  # (batch, hidden_size)

        out = self.fc(h_last)  # (batch, num_classes)
        return out

    def reset_head(self):
        """
        Reset selettivo di MLP + FC senza toccare GRU
        """
        with torch.no_grad():
            if self.use_mlp:
                for layer in self.mlp:
                    if hasattr(layer, "reset_parameters"):
                        layer.reset_parameters()
            self.fc.reset_parameters()

class MyGRUModule(nn.Module):
    def __init__(self, n_features, hidden_size=768, num_classes=2,
                 num_layers=1, use_mlp=False, mlp_hidden=768, lr_decay_factor=0.5):
        """
        n_features : int      → numero di feature in input
        hidden_size: int      → hidden size GRU
        num_classes: int      → numero di classi
        num_layers: int       → numero di layer GRU
        use_mlp   : bool      → se usare MLP tra GRU e FC
        mlp_hidden: int       → hidden size MLP
        """
        super().__init__()

        # GRU multilayer
        self.rnn = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=num_layers
        )

        # MLP opzionale prima della testa
        self.use_mlp = use_mlp
        if use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size, mlp_hidden),
                nn.ReLU(),
                #nn.Linear(mlp_hidden, hidden_size),
                #nn.ReLU()
            )

        # Testa di classificazione
        self.fc = nn.Linear(hidden_size, num_classes)
        # --- Nuovi attributi per LR decay ---
        self.optimizer = None
        self.lr_decay_factor = lr_decay_factor

    def forward(self, x):
        """
        x: tensor (batch, n_features)
        """
        # aggiungo dimensione seq_len=1
        x = x.unsqueeze(1)  # (batch, seq_len=1, n_features)
        _, h_n = self.rnn(x)  # h_n: (num_layers, batch, hidden_size)

        # prendo solo l'output dell'ultimo layer
        h_last = h_n[-1]  # (batch, hidden_size)

        # passaggio MLP se presente
        if self.use_mlp:
            h_last = self.mlp(h_last)  # (batch, hidden_size)

        out = self.fc(h_last)  # (batch, num_classes)
        return out

    def reset_head(self):
        """
        Reset selettivo di MLP + FC senza toccare GRU
        """
        with torch.no_grad():
            if self.use_mlp:
                for layer in self.mlp:
                    if hasattr(layer, "reset_parameters"):
                        layer.reset_parameters()
            self.rnn.reset_parameters()
            self.fc.reset_parameters()

    def set_optimizer(self, optimizer):
        """
        Assegna l'optimizer da usare e gestire il LR
        """
        self.optimizer = optimizer

    def reduce_lr_on_drift(self):
        """
        Decrementa il learning rate dell'optimizer al verificarsi di drift
        """
        if self.optimizer is None:
            raise ValueError("Optimizer non assegnato! Usa set_optimizer() prima.")

        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = old_lr * self.lr_decay_factor
            param_group['lr'] = new_lr
        print(f"⚠️ Drift rilevato: LR ridotto a {new_lr:.6f}")


class MyLSTMModule(nn.Module):
    def __init__(self, n_features, hidden_size=768, num_classes=2,
                 num_layers=1, use_mlp=True, mlp_hidden=768, lr_decay_factor=0.5):
        """
        n_features : int      → numero di feature in input
        hidden_size: int      → hidden size LSTM
        num_classes: int      → numero di classi
        num_layers: int      → numero di layer LSTM
        use_mlp   : bool     → se usare MLP tra LSTM e FC
        mlp_hidden: int      → hidden size MLP
        """
        super().__init__()

        # LSTM multilayer
        self.rnn = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=num_layers
        )

        # MLP opzionale
        self.use_mlp = use_mlp
        if use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size, mlp_hidden),
                nn.ReLU()
            )

        # Testa di classificazione
        self.fc = nn.Linear(hidden_size, num_classes)

        # --- LR decay ---
        self.optimizer = None
        self.lr_decay_factor = lr_decay_factor

    def forward(self, x):
        """
        x: tensor (batch, n_features)
        """
        # seq_len = 1
        x = x.unsqueeze(1)  # (batch, 1, n_features)

        _, (h_n, _) = self.rnn(x)
        # h_n: (num_layers, batch, hidden_size)

        # ultimo layer
        h_last = h_n[-1]  # (batch, hidden_size)

        if self.use_mlp:
            h_last = self.mlp(h_last)

        out = self.fc(h_last)
        return out

    def reset_head(self):
        """
        Reset selettivo di MLP + FC senza toccare LSTM
        """
        with torch.no_grad():
            if self.use_mlp:
                for layer in self.mlp:
                    if hasattr(layer, "reset_parameters"):
                        layer.reset_parameters()
            #self.rnn.reset_parameters()
            self.fc.reset_parameters()

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def reduce_lr_on_drift(self):
        if self.optimizer is None:
            raise ValueError("Optimizer non assegnato! Usa set_optimizer() prima.")

        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = old_lr * self.lr_decay_factor
            param_group['lr'] = new_lr

        print(f"⚠️ Drift rilevato: LR ridotto a {new_lr:.6f}")


class MyModule(nn.Module):
     def __init__(self, n_features):
         super().__init__()
         self.net = nn.Sequential(
             nn.Linear(n_features, n_features),
             nn.ReLU(),
             nn.Linear(n_features, 2)
         )
     def forward(self, x):
         return self.net(x)

"""
Wrapper leggero per classificatori River che espone predizione
e apprendimento online su singole istanze.
"""
class StreamClassifier:
    def __init__(self, classifier_type, params: dict):
        params = params or {}

        if classifier_type == "ARFClassifier":
            self.classifier = forest.ARFClassifier(
                drift_detector=drift.ADWIN(),seed=0
            )
        elif classifier_type == "SRPClassifier":
            base_model = tree.HoeffdingTreeClassifier()
            self.classifier = ensemble.SRPClassifier(
                model=base_model,
                drift_detector=drift.ADWIN(),seed=0
            )
        elif classifier_type == "AHFClassifier":
            self.classifier = tree.HoeffdingAdaptiveTreeClassifier(
                seed=0
            )
        
        elif classifier_type == 'NN':
            '''
            self.classifier = compose.Pipeline(
    ADWINMonitor(on='error'),
    preprocessing.StandardScaler(),
    Classifier(
        module=MyRNNModule(n_features),
        loss_fn='cross_entropy',
        optimizer_fn='adam',
        lr=1e-3,
        is_class_incremental=True
    )
)
            '''
            self.classifier =  compose.Pipeline(
     preprocessing.StandardScaler(),
     #LSTMClassifier(n_features=n_features, hidden_size=256, n_init_classes=2,
     #                      loss_fn='cross_entropy',
     #    optimizer_fn='adam',
     #    lr=1e-3,
     #    is_class_incremental=True)
     #Classifier(
    #module=BertClassifierModule(
    #    model_name="prajjwal1/bert-tiny"
    #),
    #loss_fn="cross_entropy",
    #optimizer_fn="adam",
    #lr=2e-5,
    #is_class_incremental=True
#)
    Classifier(
         module=MyLSTMModule(n_features),
         loss_fn='cross_entropy',
         optimizer_fn='rmsprop',
         lr=1e-3,
         is_class_incremental=True
     )
     
     
    
     )
     
     
        else:
            raise ValueError("Unsupported classifier type")

    def predict_one(self, x):
        return self.classifier.predict_one(x)

    def learn_one(self, x, y):
        self.classifier.learn_one(x, y)

    def predict_proba_one(self, x):
        self.classifier.predict_proba_one(x)


    def learn_many(self, x, y):
        self.classifier.learn_many(x, y)



