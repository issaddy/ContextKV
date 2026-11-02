"""Vocabulary-bound capacity of native JSON claims with required literal keys."""
from __future__ import annotations

import hashlib
import json
import re


def literal_token_coverage(text, literals, pattern):
    """Upper-bound fixed-key characters covered by one decoded vocabulary token.

    A token may contain full literals plus one fragment crossing either end,
    or lie wholly inside a literal. Counting those maxima independently can
    overcount; capping by token length still remains an upper bound.
    """
    if not text:
        return 0
    if any(text in literal for literal in literals):
        return len(text)
    complete = sum(len(match.group()) for match in pattern.finditer(text))
    limit = min(len(text), max(map(len, literals)))
    left = max((length for length in range(1, limit+1)
                if any(literal.endswith(text[:length]) for literal in literals)), default=0)
    right = max((length for length in range(1, limit+1)
                 if any(literal.startswith(text[-length:]) for literal in literals)), default=0)
    return min(len(text), complete+left+right)


def build_claim_capacity(tokenizer, required_keys, *, tokenizer_hash):
    """Use the actual vocabulary, not an encoded example's assumed minimum.

    For the native JSON grammar, each required key is emitted literally, once
    or more in each claim. If its fixed literals occupy L characters and each
    output token can cover at most M such characters, N claims need N*L <= T*M.
    Whitespace and value content can only consume more output tokens.
    """
    literals = tuple(json.dumps(key, ensure_ascii=True) for key in sorted(set(required_keys)))
    pattern = re.compile("|".join(re.escape(value) for value in sorted(literals, key=len, reverse=True)))
    ids = sorted(set(tokenizer.get_vocab().values()))
    decoded = tokenizer.batch_decode([[index] for index in ids], skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False)
    sha = hashlib.sha256()
    maximum, witness = 0, None
    for index, text in zip(ids, decoded):
        sha.update(json.dumps([index, text], ensure_ascii=True, separators=(",", ":")).encode())
        coverage = literal_token_coverage(text, literals, pattern)
        if coverage > maximum:
            maximum, witness = coverage, dict(token_id=index, decoded=text)
    return dict(kind="required_literal_vocabulary_capacity_v1", tokenizer_hash=tokenizer_hash,
        required_keys=sorted(set(required_keys)), required_literals=literals,
        required_literal_characters_per_claim=sum(map(len, literals)),
        max_token_literal_coverage=maximum, vocabulary_sha256=sha.hexdigest(),
        vocabulary_tokens=len(ids), maximum_coverage_witness=witness,
        protocol_contract="native JSON-schema required property names emitted as literal quoted keys; alternate key-escaping or relaxed-required grammar needs its own bound",
        bound="sum floor(parent_output_budget * max_token_literal_coverage / required_literal_characters_per_claim)")


def claim_count_ceiling(output_budgets, bound):
    return sum(budget*bound["max_token_literal_coverage"]//bound["required_literal_characters_per_claim"]
               for budget in output_budgets)
