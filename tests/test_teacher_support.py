import json

import httpx

from adapters.support_model import SupportModel
from adapters.teacher_support import TeacherSupportBackend
from controller.context_schema import EvidenceKey, EvidencePool, EvidenceUnit, Need, SourceSpan, TaskState
from controller.templates import ByteTemplate


def test_teacher_schema_binds_each_group_to_its_source_spans_and_records_cost():
    calls = []

    def respond(request):
        body = json.loads(request.content)
        schema = json.loads(body["sampling_params"]["json_schema"])
        calls.append(schema)
        if "predictions" in schema["properties"]:
            values = {}
            for index, item in schema["properties"]["predictions"]["properties"].items():
                permitted = item["properties"]["span_ids"]["items"]["enum"]
                assert all(span_id.startswith("s") and len(span_id) < 4 for span_id in permitted)
                values[index] = {"applicability": 1, "direct_support": .4, "joint_support": .9,
                    "bridge_value": .8, "disposition": "partial", "span_ids": permitted,
                    "missing_relation": "procedure"}
            output = {"predictions": values}
        else:
            output = {"quality": .9, "disposition": "answer"}
        return httpx.Response(200, json={"text": json.dumps(output),
                                        "meta_info": {"completion_tokens": 24}})

    backend = TeacherSupportBackend("http://teacher.test", ByteTemplate())
    backend.client.close()
    backend.client = httpx.Client(base_url="http://teacher.test", transport=httpx.MockTransport(respond))
    units = []
    for index in range(2):
        key = EvidenceKey("test", "document", str(index), "v1", str(index))
        units.append(EvidenceUnit(str(index), (SourceSpan(f"original-span-{index}", key, 4, 8),), "body"))
    model = SupportModel(backend, history_resolver=lambda ref: "")
    state = TaskState("task", "eligibility and procedure", needs=(Need("need"),))
    scores = model.score_support(state, EvidencePool(tuple(units)))
    assert len(scores.singles) == 2
    assert len(scores.bundles) == 1
    assert {span.span_id for row in scores.singles for span in row.source_spans} == {
        "original-span-0", "original-span-1"}
    assert model.predict_set(state, tuple(units), scores.singles).quality == .9
    assert backend.usage["input_tokens"] > 0
    assert backend.usage["output_tokens"] == len(calls)*24
    assert any(record["source_span_catalog"] for record in backend.records)
    model.release_task("task")
    assert not backend._sets
    assert not backend._queries
    model.close()
