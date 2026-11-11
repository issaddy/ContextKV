# Loong

Upstream: [MozerWang/Loong](https://github.com/MozerWang/Loong), commit
`6d2115b8b48a3d19412ccb52a0c9c4ee37869af4`.

Paper: Minzheng Wang and coauthors, *Leave No Document Behind: Benchmarking
Long-Context LLMs with Extended Multi-Doc QA*, EMNLP 2024.

The repository is released under [Apache License 2.0](APACHE-2.0.txt).
`bench/assets/loong/evaluation.txt` preserves its `get_evaluate_prompts` rubric.
`adapters/loong.py` adapts the source reading rules, and `bench/loong_score.py`
uses the existing local completion transport instead of the upstream provider
clients. The input adapter retains every allowed document, fixes document order
to the published list, rejects ambiguous filename matches, and recognizes legal
Link the Links questions in the actual `question` field. That task supplies fact
descriptions and judgment options separately, as described in paper §3.2.4;
appending each judgment to its source facts would change the intended task.
Legal record names can contain the requested case taxonomy. They remain offline;
model-visible source addresses and document IDs use opaque digests instead.

The scorer retains question/reference construction and the published rubric;
it uses the author's first-match score parser and a declared judge model and budgets.
Saved grades are reused when scoring resumes. Results identify the actual judge
alongside the original rubric. Input protocol
changes and source resolution failures remain in generated manifests. The public
questions and document archive retain their source identities and are stored
outside the package under `data/`.
