import pytorch_lightning as pl
import torch.nn as nn


class InferenceDNNRegressor(pl.LightningModule):
    def __init__(self, industry: bool):
        super(InferenceDNNRegressor, self).__init__()
        self.indutry = industry

        output_dim = 7

        if industry:
            input_dim = 45
            hidden_dims = [256, 128]
            use_batch_norm = False
            dropout_prob = 0.3
        else:
            input_dim = 38
            hidden_dims = [128, 64, 32]
            use_batch_norm = False
            dropout_prob = 0.0

        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.network:
            x = layer(x)
        return x
