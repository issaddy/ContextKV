"""CPU execution of pinned upstream accumulation; no scheduler or GPU simulation."""
import ast
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace


def test_mixed_finished_batch_only_publishes_terminal_request_metadata(tmp_path):
    root = Path(__file__).resolve().parents[1]
    fixtures = root/'tests/fixtures/sglang_71de97b'
    provenance = json.loads((fixtures/'source.json').read_text())
    integration = root/'engine_patches/integrations/sglang-0.5.18'
    manifest = json.loads((integration/'manifest.json').read_text())
    patch = integration/'contextkv.patch'
    assert provenance['commit'] == manifest['commit']
    assert hashlib.sha256(patch.read_bytes()).hexdigest() == manifest['patch_sha256']
    original = (fixtures/'output_streamer.txt').read_bytes()
    assert hashlib.sha256(original).hexdigest() == provenance['output_streamer']['sha256']
    target = tmp_path/provenance['output_streamer']['path']
    target.parent.mkdir(parents=True)
    target.write_bytes(original)
    # One application verifies the edited source hunk against the exact official commit.
    # A fixture under this checkout must not inherit its parent Git worktree.
    subprocess.run(['git', 'apply', '--include='+provenance['output_streamer']['path'], str(patch)],
                   cwd=tmp_path, env={**os.environ, 'GIT_CEILING_DIRECTORIES': str(tmp_path.parent)},
                   check=True, capture_output=True, text=True)
    parsed = ast.parse(target.read_text())
    native = next(node for node in parsed.body if isinstance(node, ast.ClassDef)
                  and node.name == '_GenerationStreamAccumulator')
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0), native], type_ignores=[])
    namespace = {'__name__': __name__, 'dataclass': dataclass, 'field': field}
    exec(compile(ast.fix_missing_locations(module), str(target), 'exec'), namespace)
    OutputAccumulator = namespace[native.name]

    def request(rid, done, count, info=None):
        ids = list(range(count))
        return SimpleNamespace(rid=rid, finished=lambda: done, finished_output=False,
            finished_len=None, stream=True, check_match_stop_str_prefix=lambda: False,
            sampling_params=SimpleNamespace(stream_interval=1,
                skip_special_tokens=True, spaces_between_special_tokens=True, no_stop_trim=False),
            output_ids=ids, output_ids_through_stop=ids, send_token_offset=0,
            send_output_token_logprobs_offset=0, origin_input_ids=[9],
            finished_reason=SimpleNamespace(to_json=lambda: {'type': 'stop'}) if done else None,
            http_worker_ipc='worker', decoded_text='', init_incremental_detokenize=lambda: (ids, 0),
            send_decode_id_offset=0, reasoning_tokens=0, cached_tokens=0,
            mm_image_tokens=0, mm_audio_tokens=0, mm_video_tokens=0, multimodal_inputs=None,
            retraction_count=0, time_stats=None, customized_info=info)

    def accumulator():
        return OutputAccumulator(return_logprob=False, return_hidden_states=False,
            return_routed_experts=False, return_indexer_topk=False,
            spec_algorithm=SimpleNamespace(is_none=lambda: True), disaggregation_mode=None,
            default_stream_interval=1, default_force_stream_interval=1,
            get_cached_tokens_details=lambda req: None)

    receipt = {'rid': 'terminal', 'generation_complete': True}
    timing = {'elapsed_ms': 7}
    batch = accumulator()
    for req in (request('left', False, 2), request('terminal', True, 2, {
            'contextkv_receipt': [receipt], 'contextkv_timing': [timing], 'token_score': [.1, .2]}),
            request('right', False, 3)):
        batch.accept(req=req)  # run the complete actual upstream method
    assert batch.customized_info['contextkv_receipt'] == [[], [receipt], []]
    assert batch.customized_info['contextkv_timing'] == [[], [timing], []]
    assert batch.customized_info['token_score'] == [[None, None], [.1, .2], [None, None, None]]

    fragment = (fixtures/'tokenizer_metadata.txt').read_bytes()
    assert hashlib.sha256(fragment).hexdigest() == provenance['tokenizer_metadata']['fragment_sha256']
    # Execute the real tokenizer metadata block, preserving its native per-request state.
    tokenizer_metadata = compile(ast.parse(fragment), provenance['tokenizer_metadata']['path'], 'exec')
    states = {rid: SimpleNamespace(customized_info_accumulated={}) for rid in batch.rids}

    def deliver(output):
        delivered = {}
        for i, rid in enumerate(output.rids):
            metadata = {}
            exec(tokenizer_metadata, {'state': states[rid], 'customized_info': output.customized_info,
                                     'i': i, 'meta_info': metadata})
            delivered[rid] = metadata
        return delivered

    delivered = deliver(batch)
    assert {rid for rid, info in delivered.items() if info['contextkv_receipt']} == {'terminal'}
    assert delivered['terminal']['contextkv_receipt'] == [receipt]
    assert delivered['right']['token_score'] == [None, None, None]
    # A previously unfinished request receives only its own genuine terminal metadata later.
    later = accumulator()
    own_receipt = {'rid': 'right', 'generation_complete': True}
    later.accept(req=request('right', True, 1, {
        'contextkv_receipt': [own_receipt], 'contextkv_timing': [{'elapsed_ms': 9}]}))
    assert deliver(later)['right']['contextkv_receipt'] == [own_receipt]
