"""Task heads over the shared query and evidence backbones."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

OPERATIONS = ("lookup", "enumerate", "procedure", "compare", "count", "sum", "mean", "min", "max", "group", "infer")
SCOPES = ("selected_sources", "all_supplied_documents")
SPAN_KINDS = ("need", "subject", "condition", "value", "time", "version")
CONDITION_TYPES = ("PREDICATE", "TIME", "VERSION", "PLATFORM", "LOCATION", "SUBJECT")
UPDATES = ("ADD", "REPLACE", "REMOVE", "COMPARE")
VERSION_MODES = ("current", "explicit", "as_of", "compare")
TEMPORAL_RELATIONS = ("at", "before", "after")
EDGE_KINDS = ("subject", "condition", "predecessor", "value")
FIELD_ROLES = ("source", "measure", "entity", "period", "group", "filter", "unit", "version")
FILTER_OPERATORS = ("eq", "ne", "lt", "le", "gt", "ge", "in", "contains")
STATE_DIM = 1540


class QueryHeads(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.context = nn.Linear(hidden_size, hidden_size, bias=False)
        self.span = nn.Linear(hidden_size, len(SPAN_KINDS) * 3)
        self.history = nn.Linear(hidden_size, 1)
        self.operation = nn.Linear(hidden_size, len(OPERATIONS))
        self.scope = nn.Linear(hidden_size, len(SCOPES))
        self.join = nn.Linear(hidden_size, 2)
        self.distinct = nn.Linear(hidden_size, 2)
        self.condition_type = nn.Linear(hidden_size, len(CONDITION_TYPES))
        self.update = nn.Linear(hidden_size, len(UPDATES))
        self.filter_operator = nn.Linear(hidden_size, len(FILTER_OPERATORS))
        self.version_mode = nn.Linear(hidden_size, len(VERSION_MODES))
        self.temporal_relation = nn.Linear(hidden_size, len(TEMPORAL_RELATIONS))
        self.version = nn.Bilinear(hidden_size, hidden_size, 1)
        self.family = nn.Bilinear(hidden_size, hidden_size, 1)
        self.edge = nn.Bilinear(hidden_size, hidden_size, len(EDGE_KINDS))
        self.need_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.eye_(self.need_projection.weight)

    def forward(self, token_states, pooled):
        token_states = token_states + self.context(pooled)
        return {
            "span": self.span(token_states).view(*token_states.shape[:-1], len(SPAN_KINDS), 3),
            "history": self.history(token_states).squeeze(-1),
            "version_mode": self.version_mode(pooled),
            "temporal_relation": self.temporal_relation(pooled),
        }

    def edges(self, left, right):
        a = left[:, None, :].expand(-1, right.shape[0], -1)
        b = right[None, :, :].expand(left.shape[0], -1, -1)
        return self.edge(a, b)


class EvidenceHeads(nn.Module):
    def __init__(self, hidden_size=1024, query_size=1024, embedding_size=512):
        super().__init__()
        self.query = nn.Sequential(nn.LayerNorm(query_size), nn.Linear(query_size, hidden_size, bias=False))
        self.field = nn.Linear(hidden_size, len(FIELD_ROLES) * 3)
        self.binding = nn.Bilinear(hidden_size, hidden_size, len(FIELD_ROLES))
        self.join = nn.Bilinear(hidden_size, hidden_size, 1)
        self.set_item = nn.Sequential(nn.Linear(embedding_size, 256), nn.GELU(), nn.Linear(256, 256))
        self.set_utility = nn.Sequential(nn.Linear(256 * 2 + 2, 128), nn.GELU(), nn.Linear(128, 1))

    def field_logits(self, states, need_vectors):
        queries = F.layer_norm(self.query(need_vectors), (states.shape[-1],))
        tokens = F.layer_norm(states, (states.shape[-1],))
        a, b = queries[:, None, :], tokens[None, :, :]
        # The product gives each Need its own token ordering in each BIO/role logit.
        return self.field(b + a + b*a).view(len(queries), len(states), len(FIELD_ROLES), 3)

    def utility(self, embeddings, relevance, selected):
        encoded = self.set_item(embeddings)
        mask = selected.to(encoded.dtype)
        count = mask.sum(-1, keepdim=True)
        summed = mask @ encoded
        means = summed / count.clamp_min(1)
        total_score = mask @ relevance[:, None]
        features = torch.cat((summed, means, count, total_score), dim=-1)
        return self.set_utility(features).squeeze(-1)

    def loss(self, embeddings, relevance, masks, qualities):
        predictions = self.utility(embeddings, relevance, masks)
        loss = F.mse_loss(predictions, qualities)
        pair = qualities[:, None] - qualities[None, :]
        ordered = pair > 0
        if ordered.any():
            margins = predictions[:, None] - predictions[None, :]
            loss = loss + F.softplus(-margins[ordered]).mean()
        return loss
