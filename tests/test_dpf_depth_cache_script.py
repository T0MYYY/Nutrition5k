import importlib.util
from pathlib import Path


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "generate_dpf_depth_cache.py"
    spec = importlib.util.spec_from_file_location("generate_dpf_depth_cache", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collect_depth_tasks_reports_missing_rgb(tmp_path):
    script = _load_script()
    overhead = tmp_path / "realsense_overhead"
    cache = tmp_path / "cache" / "pred_depth"
    dish_ok = overhead / "dish_0000000001"
    dish_ok.mkdir(parents=True)
    (dish_ok / "rgb.png").write_bytes(b"png")

    tasks, missing = script._collect_depth_tasks(
        ["dish_0000000001", "dish_0000000002"],
        overhead,
        cache,
    )

    assert len(tasks) == 1
    assert tasks[0][0] == "dish_0000000001"
    assert missing == ["dish_0000000002"]


def test_collect_depth_tasks_skips_existing_outputs(tmp_path):
    script = _load_script()
    overhead = tmp_path / "realsense_overhead"
    cache = tmp_path / "cache" / "pred_depth"
    dish = overhead / "dish_0000000001"
    dish.mkdir(parents=True)
    (dish / "rgb.png").write_bytes(b"png")
    out_dir = cache / "dish_0000000001"
    out_dir.mkdir(parents=True)
    (out_dir / "depth_pred.png").write_bytes(b"png")

    tasks, missing = script._collect_depth_tasks(["dish_0000000001"], overhead, cache)

    assert tasks == []
    assert missing == []
