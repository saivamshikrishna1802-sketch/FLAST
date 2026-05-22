from __future__ import annotations

import torch
from torch import nn


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.query_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.score_projection = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.score_projection(
            torch.tanh(
                self.hidden_projection(hidden_states) + self.query_projection(query).unsqueeze(1)
            )
        ).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)
        return context, weights


class PlainLSTM(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.2) -> None:
        super().__init__()
        projection_size = max(hidden_size // 2, 1)
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, projection_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_size, 1),
        )

    def forward(self, inputs: torch.Tensor, return_attention: bool = False):
        outputs, _ = self.lstm(inputs)
        prediction = self.head(self.dropout(outputs[:, -1, :])).squeeze(-1)
        if not return_attention:
            return prediction

        weights = torch.zeros(inputs.size(0), inputs.size(1), device=inputs.device)
        weights[:, -1] = 1.0
        return prediction, weights


class AttentionLSTM(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.2) -> None:
        super().__init__()
        projection_size = max(hidden_size // 2, 1)
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.attention = AdditiveAttention(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, projection_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_size, 1),
        )

    def forward(self, inputs: torch.Tensor, return_attention: bool = False):
        outputs, _ = self.lstm(inputs)
        query = outputs[:, -1, :]
        context, weights = self.attention(outputs, query)
        fused = torch.cat([context, query], dim=1)
        prediction = self.head(self.dropout(fused)).squeeze(-1)
        if not return_attention:
            return prediction
        return prediction, weights
