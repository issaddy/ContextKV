# MiniCheck adapter notice

`bench/grounding.py` uses the MiniCheck scoring and chunking implementation from
https://github.com/Liyan06/MiniCheck at commit
`b58b9fa69acbd1015ec970fa65dd752413a053d2` under Apache-2.0.

ContextKV's small loader adaptation replaces upstream model initialization with
an explicitly prepared local checkpoint, device and precision, and records
tokenization diagnostics. The upstream `MiniCheck.score` and
`Inferencer.fact_check` methods are imported, not reimplemented or vendored.
Original upstream source and pretrained weights are not included in this wheel.
The DeBERTa checkpoint's model card separately specifies MIT for its weights.

The Apache License 2.0 text is provided alongside this notice. These third-party
notices do not assign a new license to the rest of the ContextKV project.
