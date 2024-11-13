import pytorch_lightning as pl
import torch.nn as nn


class InferenceDNNRegressor(pl.LightningModule):
    def __init__(
        self, input_dim, output_dim, hidden_dims, use_batch_norm=False, dropout_prob=0.0
    ):
        super(InferenceDNNRegressor, self).__init__()

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
