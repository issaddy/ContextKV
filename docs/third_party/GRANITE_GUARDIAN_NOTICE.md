# IBM Granite Guardian attribution

`bench/granite_judge.py` reuses the native judging instruction, groundedness
criterion, scoring schema and final guardian-block format from IBM Granite
Guardian. The transport, complete-visible-evidence adaptation, input binding and
failure records are ContextKV additions; this is not the upstream evaluation suite.

Source: https://github.com/ibm-granite/granite-guardian

Fixed source commit: `0a4ec708fa916f3e9d7fc7e5b75dff836752f448`

Original authors: IBM Granite Guardian contributors.

License: Apache License, Version 2.0. The complete license is distributed beside
this notice as `APACHE-2.0.txt`. The original repository license is at
https://github.com/ibm-granite/granite-guardian/blob/0a4ec708fa916f3e9d7fc7e5b75dff836752f448/LICENSE

The separately prepared checkpoint and tokenizer are
`ibm-granite/granite-guardian-4.1-8b` at
`ab01ccca5dcfb80246369a086a4a87a29198f5af`, released under Apache-2.0.
No model weights or complete upstream repository are bundled here.
