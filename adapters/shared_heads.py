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


class ScaledBilinear(nn.Bilinear):
    def forward(self, left, right):
        left = F.layer_norm(left, (self.in1_features,))
        right = F.layer_norm(right, (self.in2_features,))
        return super().forward(left, right) * self.in1_features ** -.5

    def pairwise(self, left, right):
        """Apply each learned bilinear map once per left item, not once per pair."""
        left = F.layer_norm(left, (self.in1_features,))
        right = F.layer_norm(right, (self.in2_features,))
        projected = torch.einsum("ni,oij->noj", left, self.weight)
        return (torch.einsum("noj,mj->nmo", projected, right)+self.bias) * self.in1_features ** -.5


class QueryHeads(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.context = nn.Linear(hidden_size, hidden_size, bias=False)
        self.span = nn.Linear(hidden_size, len(SPAN_KINDS) * 3)
        self.history = nn.Linear(hidden_size, 1)
        self.operation = nn.Linear(hidden_size, len(OPERATIONS))
        self.scope = nn.Linear(hidden_size, len(SCOPES))
        self.coverage = nn.Linear(hidden_size, len(SCOPES))
        self.join = nn.Linear(hidden_size, 2)
        self.distinct = nn.Linear(hidden_size, 2)
        self.condition_type = nn.Linear(hidden_size, len(CONDITION_TYPES))
        self.update = nn.Linear(hidden_size, len(UPDATES))
        self.filter_operator = nn.Linear(hidden_size, len(FILTER_OPERATORS))
        self.version_mode = nn.Linear(hidden_size, len(VERSION_MODES))
        self.temporal_relation = nn.Linear(hidden_size, len(TEMPORAL_RELATIONS))
        self.version = ScaledBilinear(hidden_size, hidden_size, 1)
        self.family = ScaledBilinear(hidden_size, hidden_size, 1)
        self.edge = ScaledBilinear(hidden_size, hidden_size, len(EDGE_KINDS))
        self.need_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.eye_(self.need_projection.weight)

    def contextualize(self, token_states, pooled):
        tokens = F.layer_norm(token_states, (token_states.shape[-1],))
        current = self.context(F.layer_norm(pooled, (pooled.shape[-1],)))
        current = F.layer_norm(current, (current.shape[-1],))
        return F.layer_norm(tokens + current + tokens * current, (tokens.shape[-1],))

    def forward(self, token_states, pooled):
        token_states = self.contextualize(token_states, pooled)
        pooled = F.layer_norm(pooled, (pooled.shape[-1],))
        return {
            "features": token_states,
            "span": self.span(token_states).view(*token_states.shape[:-1], len(SPAN_KINDS), 3),
            "history": self.history(token_states).squeeze(-1),
            "version_mode": self.version_mode(pooled),
            "temporal_relation": self.temporal_relation(pooled),
        }

    def edges(self, left, right):
        return self.edge.pairwise(left, right)


class EvidenceHeads(nn.Module):
    def __init__(self, hidden_size=1024, query_size=1024):
        super().__init__()
        self.query = nn.Sequential(nn.LayerNorm(query_size), nn.Linear(query_size, hidden_size, bias=False))
        self.field = nn.Linear(hidden_size, len(FIELD_ROLES) * 3)
        self.binding = ScaledBilinear(hidden_size, hidden_size, len(FIELD_ROLES))
        self.join = ScaledBilinear(hidden_size, hidden_size, 1)
        self.set_weights = nn.Parameter(torch.zeros(6))
        self.set_bias = nn.Parameter(torch.tensor(-1.0))

    def field_logits(self, states, need_vectors):
        queries = F.layer_norm(self.query(need_vectors), (states.shape[-1],))
        tokens = F.layer_norm(states, (states.shape[-1],))
        a, b = queries[:, None, :], tokens[None, :, :]
        # The product gives each Need its own token ordering in each BIO/role logit.
        return self.field(b + a + b*a).view(len(queries), len(states), len(FIELD_ROLES), 3)

    @staticmethod
    def features(embeddings, relevance, selected, support, completed):
        """Bounded information state shared by selection and stopping."""
        selected = selected.bool()
        mask = selected.to(embeddings.dtype)
        count = mask.sum(-1, keepdim=True)
        maximum = relevance[None].expand_as(mask).masked_fill(~selected, -torch.inf).amax(-1, keepdim=True)
        minimum = relevance[None].expand_as(mask).masked_fill(~selected, torch.inf).amin(-1, keepdim=True)
        maximum = torch.where(count > 0, maximum, 0.)
        minimum = torch.where(count > 0, minimum, 0.)
        total = mask @ relevance[:, None]
        similarity = (embeddings @ embeddings.T).clamp(0, 1)
        order = relevance.argsort(descending=True)
        ordered = mask[:, order]
        previous = torch.tril(similarity[order][:, order], diagonal=-1)
        expressed = (previous[None]*ordered[:, None, :]).amax(-1)
        novel = (ordered*relevance[order].clamp_min(0)*(1-expressed)).sum(-1, keepdim=True)
        pairs = mask[:, :, None]*mask[:, None, :]*(1-torch.eye(len(relevance), device=mask.device))
        redundancy = (pairs*similarity).amax((1, 2))[:, None]
        if support.shape[1]:
            grounded = support*relevance.clamp_min(0)[:, None]
            need_coverage = torch.maximum((mask[:, :, None]*grounded[None]).amax(1), completed[None])
            all_coverage = torch.maximum(grounded.amax(0), completed)
            need_mean = need_coverage.mean(-1, keepdim=True)
            need_min = need_coverage.amin(-1, keepdim=True)
            gap = (all_coverage[None]-need_coverage).mean(-1, keepdim=True)
            known = completed.mean().expand_as(count)
        else:
            need_mean = need_min = known = gap = torch.zeros_like(count)
        return torch.cat((maximum, total/count.clamp_min(1), minimum,
            1-(-novel).exp(), redundancy, maximum-minimum, need_mean, need_min, known, gap,
            relevance.max().clamp_min(0)-maximum.clamp_min(0), (count > 0).to(mask.dtype)), -1)

    def utility(self, embeddings, relevance, selected, support, completed):
        """Calibrated information gain: duplicates add no positive information."""
        features = self.features(embeddings, relevance, selected, support, completed)
        positive = features[:, [0, 3, 6, 7]].clamp_min(0)
        negative = features[:, [4, 5]]
        weights = F.softplus(self.set_weights.to(embeddings.device))
        bias = self.set_bias.to(embeddings.device)
        values = (bias + positive@weights[:4] - negative@weights[4:]).sigmoid()
        prior = (completed.mean()*weights[2] + completed.min()*weights[3]
                 if len(completed) else weights.new_zeros(()))
        return values-(bias+prior).sigmoid()

    def loss(self, embeddings, relevance, masks, qualities, support, completed):
        predictions = self.utility(embeddings, relevance, masks, support, completed)
        empty = (masks == 0).all(dim=-1)
        gains = qualities-qualities[empty].mean()
        loss = F.mse_loss(predictions, gains)
        return loss + F.kl_div(predictions.log_softmax(-1), gains.softmax(-1), reduction="sum")
