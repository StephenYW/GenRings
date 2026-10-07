"""Prompt lab: sources, prompt versions (locked once run), runs and reviews, all as files."""
import csv
import io
import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import config, enhance, prompt_lab, relief_prompts
from app.main import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROMPT_LAB_DIR", tmp_path / "lab")
    monkeypatch.setattr(config, "RELIEF_PROVIDER", "mock")
    monkeypatch.setattr(config, "USAGE_LOG", tmp_path / "usage.jsonl")
    monkeypatch.setattr(enhance, "estimate_depth",
                        lambda rgb: np.linspace(0, 1, rgb.shape[0] * rgb.shape[1], dtype=np.float32).reshape(rgb.shape[:2]))
    return TestClient(app)


def _png(color=(200, 80, 40), size=(120, 90), mode="RGB"):
    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, format="PNG")
    return buf.getvalue()


def _upload(client, category, n=1, split="auto", **form):
    files = [("files", (f"img{i}.png", _png(), "image/png")) for i in range(n)]
    res = client.post("/api/lab/sources", files=files, data={"category": category, "split": split, **form})
    assert res.status_code == 200, res.text
    return res.json()


def _wait(client, run_id, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        run = client.get(f"/api/lab/runs/{run_id}").json()
        if not run["active"]:
            return run
        time.sleep(0.05)
    raise AssertionError("run didn't finish")


def test_sources_ids_auto_split_and_csv(client):
    out = _upload(client, "portrait", 5, licence="own photo")
    ids = [r["id"] for r in out["added"]]
    assert ids == [f"portrait-{i:03d}" for i in range(1, 6)]
    assert [r["split"] for r in out["added"]] == ["tune", "tune", "tune", "holdout", "tune"]  # every 4th held out
    lab = config.PROMPT_LAB_DIR
    assert (lab / "sources" / "portrait" / "portrait-001.png").exists()
    assert (lab / "README.md").exists() and (lab / "error_tags.json").exists()
    rows = list(csv.DictReader(open(lab / "sources" / "sources.csv")))
    assert rows[0]["licence"] == "own photo" and rows[0]["origin"] == "img0.png"

    bad = client.post("/api/lab/sources", files=[("files", ("x.txt", b"hello", "text/plain"))],
                      data={"category": "logo"}).json()
    assert bad["added"] == [] and "not a PNG" in bad["errors"][0]
    assert client.post("/api/lab/sources", files=[("files", ("a.png", _png(), "image/png"))],
                       data={"category": "selfie"}).status_code == 400

    assert client.patch("/api/lab/sources/portrait-002", json={"split": "holdout", "notes": " blurry "}).json()["notes"] == "blurry"
    client.delete("/api/lab/sources/portrait-001")
    assert [r["id"] for r in client.get("/api/lab/sources").json()["sources"]][0] == "portrait-002"
    assert not (lab / "sources" / "portrait" / "portrait-001.png").exists()


def test_prompt_versions_build_and_lock(client):
    versions = client.get("/api/lab/prompts").json()["versions"]
    assert [v["version"] for v in versions] == ["v1"] and not versions[0]["locked"]
    v1 = client.get("/api/lab/prompts/v1").json()
    assert v1["templates"]["convert"] == relief_prompts.CONVERT
    assert relief_prompts.IMAGE_TYPES["drawing"].prompt in v1["examples"]["drawing"]

    v2 = client.post("/api/lab/prompts", json={"based_on": "v1", "notes": "shorter casting"}).json()
    assert v2["version"] == "v2" and v2["based_on"] == "v1"
    templates = {**v2["templates"], "casting": "Keep it simple for casting."}
    templates["image_types"] = {**templates["image_types"], "logo": "LOGO SENTENCE."}
    saved = client.put("/api/lab/prompts/v2", json={"notes": "shorter", "settings": v2["settings"],
                                                    "templates": templates}).json()
    assert "Keep it simple for casting." in saved["examples"]["portrait"]
    assert "LOGO SENTENCE." in saved["examples"]["logo"]
    # a stray placeholder is refused
    broken = {**templates, "convert": "Convert into {material}"}
    res = client.put("/api/lab/prompts/v2", json={"notes": "", "settings": v2["settings"], "templates": broken})
    assert res.status_code == 400 and "placeholder" in res.json()["detail"]

    _upload(client, "logo", 1)
    run = client.post("/api/lab/runs", json={"version": "v2", "split": "all"}).json()
    _wait(client, run["run_id"])
    assert client.get("/api/lab/prompts/v2").json()["locked"]
    res = client.put("/api/lab/prompts/v2", json={"notes": "", "settings": v2["settings"], "templates": templates})
    assert res.status_code == 409


def test_run_generates_files_and_reviews_summarise(client):
    _upload(client, "portrait", 4)          # portrait-004 is holdout
    _upload(client, "animal", 2)
    run = client.post("/api/lab/runs", json={"version": "v1", "split": "tune"}).json()
    run = _wait(client, run["run_id"])
    assert run["run_id"].endswith("_v1")
    ids = [i["source_id"] for i in run["items"]]
    assert ids == ["portrait-001", "portrait-002", "portrait-003", "animal-001", "animal-002"]
    assert all(i["status"] == "done" for i in run["items"])

    d = config.PROMPT_LAB_DIR / "runs" / run["run_id"] / "portrait-001"
    for f in ("generation.webp", "depth.png", "relief.webp", "meta.json"):
        assert (d / f).exists(), f
    meta = json.loads((d / "meta.json").read_text())
    assert meta["seed"] == prompt_lab.seed_for("portrait-001") and meta["version"] == "v1"
    assert relief_prompts.IMAGE_TYPES["portrait"].prompt in meta["prompt"]

    rid = run["run_id"]
    url = f"/api/lab/runs/{rid}/items"
    assert client.put(f"{url}/portrait-001/review", json={"verdict": "bad", "score": 2,
                                                         "errors": ["identity_changed", "identity_changed"],
                                                         "comment": "nose"}).status_code == 200
    client.put(f"{url}/portrait-002/review", json={"verdict": "good", "score": 5})
    client.put(f"{url}/animal-001/review", json={"verdict": "bad", "score": 1, "errors": ["identity_changed"]})
    assert client.put(f"{url}/portrait-004/review", json={"verdict": "good"}).status_code == 404  # not in this run
    assert client.put(f"{url}/portrait-001/review", json={"verdict": "meh"}).status_code == 422

    summary = json.loads((config.PROMPT_LAB_DIR / "runs" / rid / "summary.json").read_text())
    assert summary["reviewed"] == 3 and summary["verdicts"] == {"good": 1, "usable": 0, "bad": 2}
    assert summary["errors"] == {"identity_changed": 2} and summary["avg_score"] == round(8 / 3, 2)
    assert summary["by_category"]["portrait"]["avg_score"] == 3.5
    assert summary["by_category"]["animal"]["verdicts"]["bad"] == 1

    # clearing a review removes it
    client.put(f"{url}/portrait-002/review", json={})
    assert not (config.PROMPT_LAB_DIR / "runs" / rid / "portrait-002" / "review.json").exists()

    # a second run of the same version on the same day gets its own folder
    run2 = client.post("/api/lab/runs", json={"version": "v1", "source_ids": ["portrait-004"]}).json()
    assert run2["run_id"] == rid + "_2"
    _wait(client, run2["run_id"])
    assert [r["run_id"] for r in client.get("/api/lab/runs").json()["runs"]][0] == rid + "_2"


def test_failed_items_can_be_resumed(client, monkeypatch):
    _upload(client, "landscape", 2)
    calls = {"n": 0}

    def flaky_generate(self, rgb, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("model hiccup")
        return orig(self, rgb, *a, **k)

    from app import relief
    orig = relief.MockRelief.generate
    monkeypatch.setattr(relief.MockRelief, "generate", flaky_generate)
    monkeypatch.setattr(config, "PROMPT_LAB_PARALLEL", 1)
    run = client.post("/api/lab/runs", json={"version": "v1", "split": "all"}).json()
    run = _wait(client, run["run_id"])
    assert [i["status"] for i in run["items"]] == ["failed", "done"]
    assert run["items"][0]["error"] == "model hiccup"
    client.post(f"/api/lab/runs/{run['run_id']}/resume", json={})
    run = _wait(client, run["run_id"])
    assert [i["status"] for i in run["items"]] == ["done", "done"] and calls["n"] == 3


def test_transparent_logo_goes_on_white_and_files_are_confined(client):
    buf = io.BytesIO()
    Image.new("RGBA", (40, 30), (0, 0, 0, 0)).save(buf, format="PNG")
    res = client.post("/api/lab/sources", files=[("files", ("logo.png", buf.getvalue(), "image/png"))],
                      data={"category": "logo"}).json()
    rgb = prompt_lab.load_rgb(config.PROMPT_LAB_DIR / "sources" / res["added"][0]["file"])
    assert rgb.shape == (30, 40, 3) and rgb.min() == 255

    assert client.get("/api/lab/files/sources/logo/logo-001.png").status_code == 200
    thumb = client.get("/api/lab/files/sources/logo/logo-001.png?w=128")
    assert thumb.status_code == 200 and thumb.headers["content-type"] == "image/webp"
    assert client.get("/api/lab/files/../../app/config.py").status_code == 404
    assert client.get("/api/lab/files/%2e%2e/%2e%2e/app/config.py").status_code == 404


def test_custom_error_tags(client):
    tags = client.post("/api/lab/tags", json={"group": "Accuracy", "label": "Wrong eye colour"}).json()
    assert tags["tag"] == "wrong_eye_colour"
    assert "wrong_eye_colour" in next(g for g in tags["tags"] if g["group"] == "Accuracy")["tags"]
    new = client.post("/api/lab/tags", json={"group": "Ring", "label": "Too busy"}).json()
    assert {"group": "Ring", "tags": ["too_busy"]} in new["tags"]
    assert client.get("/api/lab/options").json()["tags"] == new["tags"]
