"""Optional Qwen3 tokenizer contracts use local metadata only, never weights."""
import os
from pathlib import Path

import pytest


@pytest.fixture
def qwen3_tokenizer_template():
    if os.environ.get('CONTEXTKV_TEST_HF_TOKENIZER') != '1':
        pytest.skip('opt-in local Qwen3 tokenizer contract')
    path = os.environ.get('CONTEXTKV_TEST_TOKENIZER_PATH')
    if not path:
        pytest.fail('set CONTEXTKV_TEST_TOKENIZER_PATH to the local Qwen3-32B tokenizer metadata')
    from controller.templates import HuggingFaceTemplate
    from engine_patches.contextkv_model_scope import load_model_scope
    from dataclasses import asdict
    scope = load_model_scope(Path(__file__).resolve().parents[1] / 'configs/model_scopes/qwen3-32b.local.json')
    template = HuggingFaceTemplate(path, None, model_identity=scope.identity, local_files_only=True)
    scope.verify_fingerprint(asdict(template.fingerprint))
    return template
