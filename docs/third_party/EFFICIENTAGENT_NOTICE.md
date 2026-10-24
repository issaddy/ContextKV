# EfficientAgent host admission attribution

ContextKV's optional `HostAdmissionPolicy` adapts the small write-admission predicate and activity/pressure window idea from EfficientAgent. The original implementation is available in the authors' repository:

- Repository: https://github.com/KunmingSHAO/efficientagent_release
- Fixed revision: `b21b12a07a3ccf473c82acf00820b1aa2073044b`
- Source: https://github.com/KunmingSHAO/efficientagent_release/blob/b21b12a07a3ccf473c82acf00820b1aa2073044b/efficientagent/kvtier/connector.py
- Source SHA256: `c4c7eea4dd16b227548cf04330a69baa686ea1cdb7c9f9a82bc7da4595320268`
- License: Apache License 2.0, https://github.com/KunmingSHAO/efficientagent_release/blob/b21b12a07a3ccf473c82acf00820b1aa2073044b/LICENSE
- The complete license is retained in [APACHE-2.0.txt](APACHE-2.0.txt), byte-identical SHA256 `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`.

Changes made by ContextKV, 2026-10-24: the policy uses SGLang FULL `BackupKV` action pages, existing native request/business/epoch identities, same-process allocator occupancy and allocation-triggered host eviction counters. It does not reuse the LMCache/vLLM connector, chunk-key serialization, worker wrapper, background telemetry thread, or model/TP-specific byte constants. The 8192-token starting threshold corresponds to the upstream default of eight 1024-token chunks, rounded to the actual page unit; it is not a validated optimal parameter. ContextKV extends the predicate to an incremental per-request backup budget: only FULL pages whose native host allocation, commit and pending tracking succeeded spend the budget; each action is accepted or skipped as a whole before allocation. Failed allocations and already-backed zero-page specs spend nothing. This differs from upstream once-per-request prompt-chunk admission and must be evaluated separately. This adaptation does not inherit the original paper's performance claims.
