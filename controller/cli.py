"""Serve the current ContextKV configuration or inspect committed history."""
from __future__ import annotations

import argparse
from pathlib import Path

from .conversation_store import ConversationStore
from .serialization import dumps


def main(argv=None):
    parser = argparse.ArgumentParser(prog="contextkv")
    parser.add_argument("--config", default="configs/study.yaml")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("session_id")
    inspect.add_argument("--state-dir", required=True)
    args = parser.parse_args(argv)
    if args.command == "serve":
        import uvicorn
        from .api import create_app
        uvicorn.run(create_app(study_config=args.config), host=args.host, port=args.port)
    else:
        store = ConversationStore(Path(args.state_dir) / "conversations.sqlite3")
        try:
            print(dumps(store.get_snapshot(args.session_id).raw_history))
        finally:
            store.close()


if __name__ == "__main__":
    main()
