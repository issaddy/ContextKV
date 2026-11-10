"""Service entry point and persisted-session management."""
from __future__ import annotations
import argparse
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import yaml
from .config import RunConfig, create_controller, create_model_template, load_config
from .serialization import dumps, to_jsonable
from .session_store import SessionStore
from .templates import ByteTemplate


def main(argv=None):
    parser = argparse.ArgumentParser(prog="contextkv", description="ContextKV service and session management")
    parser.add_argument("--config", default="configs/qwen3_32b.yaml", help="YAML configuration")
    parser.add_argument("--state-dir", help="Override persistent output directory")
    prompt = parser.add_mutually_exclusive_group()
    prompt.add_argument("--system-prompt", help="Override context.system_prompt for this run")
    prompt.add_argument("--system-prompt-file", help="UTF-8 system prompt; full content is recorded in run configuration")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="Start the local controller API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    retrieval = serve.add_mutually_exclusive_group()
    retrieval.add_argument("--retriever-index", help="Frozen BM25S index for HTTP query retrieval")
    retrieval.add_argument("--retriever-config", help="Shared hybrid retriever YAML/JSON for HTTP queries")
    retrieval.add_argument("--study-config", help="Task/support/reading study configuration")
    serve.add_argument("--evidence-corpus", help="Optional original version/condition metadata JSONL")
    serve.add_argument("--corpus-id", default="mtrag", help="Evidence corpus namespace; snapshot comes from the index")
    inspect = sub.add_parser("inspect", help="Read a persisted session")
    inspect.add_argument("session_id")
    recover = sub.add_parser("recover", help="Recover a pending engine result, or explicitly abandon preparation")
    recover.add_argument("session_id")
    recover.add_argument("--abandon", action="store_true")
    sub.add_parser("fingerprint", help="Print the model/template fingerprint for the engine patch")
    args = parser.parse_args(argv)
    if args.command == "inspect":
        if args.state_dir:
            state_dir = args.state_dir
        else:
            settings = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
            state_dir = settings.get("state_dir", RunConfig.state_dir)
        store = SessionStore(Path(state_dir) / "sessions.sqlite3")
        try:
            print(json.dumps(to_jsonable(store.get_snapshot(args.session_id)), indent=2, ensure_ascii=False))
        finally:
            store.close()
        return
    config = load_config(args.config)
    if args.command == "fingerprint":
        template = (ByteTemplate(namespace_hash=config.model.namespace_hash) if config.engine.kind == "mock"
                    else create_model_template(config.model))
        print(dumps(template.fingerprint))
        return
    if args.state_dir:
        config = replace(config, state_dir=args.state_dir)
    system_prompt = args.system_prompt
    if args.system_prompt_file:
        with Path(args.system_prompt_file).open(encoding="utf-8", newline="") as stream:
            system_prompt = stream.read()
    if system_prompt is not None:
        config = replace(config, context=replace(config.context, system_prompt=system_prompt))
    if args.command == "serve":
        import uvicorn
        from .api import create_app
        if args.evidence_corpus and not (args.retriever_index or args.retriever_config):
            parser.error("--evidence-corpus requires --retriever-index or --retriever-config")
        uvicorn.run(create_app(config=config, retriever_index=args.retriever_index,
            retriever_config=args.retriever_config, study_config=args.study_config,
            evidence_corpus=args.evidence_corpus,
            corpus_id=args.corpus_id), host=args.host, port=args.port)
    else:
        service = create_controller(config)
        async def recover_pending():
            try:
                result = await service.recover(args.session_id, abandon=args.abandon)
                print(dumps(result))
            finally:
                await service.aclose()
        asyncio.run(recover_pending())
        service.store.close()

if __name__ == "__main__":
    main()
