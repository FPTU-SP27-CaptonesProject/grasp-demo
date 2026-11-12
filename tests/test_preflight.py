from grasp_demo.preflight import judge, pick_questions


def question(seconds, grasp=10.0, match=True):
    columns = {"rag_7b": {"seconds": 3.0}, "grasp_sft": {"seconds": grasp}, "grasp_grpo": {"seconds": grasp}}
    return {"seconds": seconds, "columns": columns, "grpo_matches_report": match}


def result(**overrides):
    base = {"health": {"reader": {"ready": True}, "retriever": {"ready": True}, "proxy": {"ready": True}, "strong": {"ready": False}},
            "vram_peak_gb": 14.2, "questions": [question(30.0) for _ in range(6)]}
    base.update(overrides)
    return base


def test_judge_passes_when_every_threshold_holds_and_ignores_bedrock():
    verdict = judge(result())
    assert verdict == {"health": True, "vram": True, "speed": True, "grpo_matches": True, "local_ok": True}


def test_judge_fails_on_vram_speed_or_matches():
    assert judge(result(vram_peak_gb=15.5))["local_ok"] is False
    assert judge(result(questions=[question(61.0)] + [question(30.0)] * 5))["speed"] is False
    assert judge(result(questions=[question(30.0, grasp=41.0)] * 6))["speed"] is False
    assert judge(result(questions=[question(30.0, match=False)] * 3 + [question(30.0)] * 3))["grpo_matches"] is False


def test_errored_grasp_column_fails_speed_and_missing_report_is_unknown():
    broken = question(30.0)
    broken["columns"]["grasp_grpo"] = {"error": "boom"}
    assert judge(result(questions=[broken] * 6))["speed"] is False
    unknown = judge(result(questions=[question(30.0, match=None)] * 6))
    assert unknown["grpo_matches"] is None and unknown["local_ok"] is False


def test_pick_two_questions_per_dataset_in_file_order():
    rows = [{"id": f"{d}{i}", "metadata": {"dataset": d}} for d in ("a", "b") for i in range(3)]
    assert [row["id"] for row in pick_questions(rows)] == ["a0", "a1", "b0", "b1"]
