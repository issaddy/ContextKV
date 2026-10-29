"""Declared model revision and tokenization shared by serving and the native engine."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys


class LocalModelScope:
    def __init__(self, record, *, source_path):
        self._record = deepcopy(record)
        self.identity = record["revision"]
        self.model_path = str(Path(record["model_path"]).resolve())
        self._record["model_path"] = self.model_path
        self.context_limit = record["config"]["max_position_embeddings"]
        self.source_path = str(source_path)

    def manifest(self):
        return {"source_path": self.source_path, "identity": self.identity,
                "record": deepcopy(self._record)}

    def verify_files(self, model_path=None):
        path = Path(model_path or self.model_path).resolve()
        if str(path) != self.model_path:
            raise ValueError("running model path differs from local scope")
        if json.loads((path/"config.json").read_text()) != self._record["config"]:
            raise ValueError("local model config differs from scope")

    def verify_fingerprint(self, fingerprint):
        expected = {"model_revision": self.identity, "tokenizer_hash": self._record["tokenizer_hash"],
                    "template_hash": self._record["template_hash"]}
        if any(fingerprint.get(key) != value for key, value in expected.items()):
            raise ValueError("model scope and client fingerprint differ")

    def verify_tokenizer(self, tokenizer):
        actual = hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
        template = json.dumps({"template": tokenizer.chat_template, "enable_thinking": False}, sort_keys=True)
        if (actual != self._record["tokenizer_hash"]
                or hashlib.sha256(template.encode()).hexdigest() != self._record["template_hash"]):
            raise ValueError("running tokenizer/nonthinking template differs from scope")

    def verify_native_settings(self, settings):
        if (settings.revision is not None or settings.dtype != "bfloat16" or settings.kv_cache_dtype != "bfloat16"
                or settings.quantization is not None or settings.load_format == "dummy"
                or json.loads(settings.json_model_override_args or "{}")):
            raise ValueError("local scope requires actual checkpoint loading, BF16 weights/KV, no revision, quantization or model override")
        if (Path(settings.model_path).resolve() != Path(self.model_path)
                or Path(settings.tokenizer_path).resolve() != Path(self.model_path)):
            raise ValueError("native model/tokenizer paths differ from local scope")
        if settings.context_length is None or not 0 < settings.context_length <= self.context_limit:
            raise ValueError("native context length exceeds the scoped model configuration")


def load_model_scope(path):
    path = Path(path).resolve()
    return LocalModelScope(json.loads(path.read_text()), source_path=path)


def launch_argv(scope, arguments):
    # These values are owned by the explicit record, not trailing overrides.
    reserved = {"--model-path", "--tokenizer-path", "--revision"}
    if any(item.split("=", 1)[0] in reserved for item in arguments):
        raise ValueError("model/tokenizer/revision launch values are supplied only by model scope")
    declared_path = os.environ.get("CONTEXTKV_MODEL_PATH")
    if declared_path and Path(declared_path).resolve() != Path(scope.model_path):
        raise ValueError("CONTEXTKV_MODEL_PATH conflicts with explicit model scope")
    return [sys.executable, "-m", "sglang.launch_server", "--model-path", scope.model_path,
            "--tokenizer-path", scope.model_path, *arguments]

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--launch", action="store_true")
    args, remaining = parser.parse_known_args(argv)
    scope = load_model_scope(args.scope)
    if not args.launch:
        if remaining:
            parser.error("unexpected arguments without --launch")
        print(json.dumps(scope.manifest(), indent=2))
        return 0
    scope.verify_files()
    scope.verify_fingerprint(json.loads(os.environ["CONTEXTKV_FINGERPRINT_JSON"]))
    arguments = launch_argv(scope, remaining)
    os.environ["CONTEXTKV_MODEL_SCOPE_PATH"] = scope.source_path
    os.execv(sys.executable, arguments)

if __name__ == "__main__":
    raise SystemExit(main())
