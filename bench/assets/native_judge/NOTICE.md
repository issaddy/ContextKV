# Native judge prompt attribution

`templates.json` contains unmodified prompt strings from:

- Atla's Selene Mini, Apache-2.0, commit
  `418c5ce2d8c4d89c5bd971ce60c7589eb8f9d8de`:
  `prompt-templates/absolute-scoring.yaml` and
  `prompt-templates/absolute-scoring-with-reference.yaml`.
  Source: https://github.com/atla-ai/selene-mini/tree/418c5ce2d8c4d89c5bd971ce60c7589eb8f9d8de
- Prometheus Eval, Apache-2.0, commit
  `dcfb44272d5d0414832f5dbb8c2a05ebc2614234`:
  `libs/prometheus-eval/prometheus_eval/prompts.py`, selecting
  `ABS_SYSTEM_PROMPT`, `ABSOLUTE_PROMPT`, and `ABSOLUTE_PROMPT_WO_REF`.
  Source: https://github.com/prometheus-eval/prometheus-eval/tree/dcfb44272d5d0414832f5dbb8c2a05ebc2614234

The original prompt strings are stored as JSON rather than YAML/Python. The
project-specific five criteria, data-boundary instructions, optional NA extension,
normalization and parser are ContextKV adaptations, not claims about the authors'
original evaluation protocol. No model weights are included. See `LICENSE.txt`.
