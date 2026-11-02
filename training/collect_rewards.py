"""Collect frozen-reader answer preferences and retain complete request costs."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import yaml


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text())
    if args.limit is not None:
        config["training"]["preference_turns"] = args.limit
    from training.preference_rollout import collect_preferences
    print(json.dumps(asyncio.run(collect_preferences(config)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
