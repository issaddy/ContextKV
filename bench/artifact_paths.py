"""Explicit archive relocation; paths locate artifacts, digests identify bytes."""
from pathlib import Path


def remap(value, mapping):
    if isinstance(value, dict):
        return {key: remap(item, mapping) for key, item in value.items() if key != "path_map"}
    if isinstance(value, list):
        return [remap(item, mapping) for item in value]
    if isinstance(value, str):
        for old, new in sorted(mapping.items(), key=lambda pair: -len(pair[0])):
            if value == old or value.startswith(old.rstrip("/") + "/"):
                return str(Path(new) / value[len(old):].lstrip("/"))
    return value
