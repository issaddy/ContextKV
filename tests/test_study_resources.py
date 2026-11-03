from bench.study_resources import attribute_physical, integrate_physical


def test_shared_physical_memory_is_charged_once_across_overlapping_requests():
    samples = [{"timestamp": float(t), "gpu_gib": 10.0, "host_gib": 2.0} for t in range(4)]
    costs = attribute_physical(samples, {"a": (0, 2), "b": (1, 3)})
    total = integrate_physical(samples, 0, 3)
    assert costs["a"]["gpu_gib_seconds"] == 15
    assert costs["b"]["gpu_gib_seconds"] == 15
    assert sum(row["gpu_gib_seconds"] for row in costs.values()) == total["gpu_gib_seconds"] == 30
    assert sum(row["host_gib_seconds"] for row in costs.values()) == total["host_gib_seconds"] == 6


def test_idle_residency_is_retained_in_batch_integral():
    samples = [{"timestamp": float(t), "gpu_gib": 10.0, "host_gib": 2.0} for t in range(5)]
    costs = attribute_physical(samples, {"a": (0, 1), "b": (3, 4)})
    assert sum(row["gpu_gib_seconds"] for row in costs.values()) == 20
    assert integrate_physical(samples, 0, 4)["gpu_gib_seconds"] == 40
