import json

from bench.study import save_measurement_segment


def test_resumed_measurements_preserve_each_raw_series_and_idle_batch_cost(tmp_path):
    first = save_measurement_segment(tmp_path, "segment-one", physical={"samples": [1, 2]},
        native={"seq": [10, 11]}, batch={"gpu_gib_seconds": 20, "idle_seconds": 5})
    before = {path.name: path.read_bytes() for path in first.iterdir()}
    second = save_measurement_segment(tmp_path, "segment-two", physical={"samples": [3, 4]},
        native={"seq": [12, 13]}, batch={"gpu_gib_seconds": 30, "idle_seconds": 7})
    assert {path.name: path.read_bytes() for path in first.iterdir()} == before
    index = json.loads((tmp_path/"measurement_segments.json").read_text())
    assert set(index["segments"]) == {str(first.relative_to(tmp_path)), str(second.relative_to(tmp_path))}
    assert sum(json.loads((tmp_path/path/"batch_cost.json").read_text())["idle_seconds"]
               for path in index["segments"]) == 12
