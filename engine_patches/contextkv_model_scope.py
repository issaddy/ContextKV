"""Declared model revision and tokenization shared by serving and the native engine."""
from __future__ import annotations

import argparse
from copy import deepcopy
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

def load_model_scope(path):
    path = Path(path).resolve()
    return LocalModelScope(json.loads(path.read_text()), source_path=path)


def launch_argv(scope, arguments):
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
    arguments = launch_argv(scope, remaining)
    os.environ["CONTEXTKV_MODEL_SCOPE_PATH"] = scope.source_path
    os.execv(sys.executable, arguments)

if __name__ == "__main__":
    raise SystemExit(main())
