# coding=utf-8
"""
test_mine_logics_api.py - /api/mine-logics REST API 测试 (v3.0.3 Step 2)

覆盖:
- POST /start → 200 + run_id
- GET /status/{run_id} → 200 + progress
- GET /results/{run_id} → 200 + result (or pending)
- POST /stop/{run_id} → 200 + stopped
- GET /history → 200 + runs list
- GET /status/nonexistent → 404
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


# ======================================================================
# REST Endpoints
# ======================================================================
class TestMineLogicsStart:
    def test_start_returns_200(self):
        resp = client.post("/api/mine-logics/start", json={
            "source_libs": ["alpha101"],
            "max_per_lib": 1,
            "workers": 1,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "run_id" in data
        assert data["run_id"].startswith("ml-")
        assert data["status"] == "pending"

    def test_start_with_defaults(self):
        resp = client.post("/api/mine-logics/start", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"].startswith("ml-")

    def test_start_invalid_returns_422(self):
        resp = client.post("/api/mine-logics/start", json={
            "max_per_lib": -1,
        })
        assert resp.status_code == 422


class TestMineLogicsStatus:
    def test_status_returns_200(self):
        start_resp = client.post("/api/mine-logics/start", json={
            "source_libs": ["alpha101"],
            "max_per_lib": 1,
            "workers": 1,
        })
        run_id = start_resp.json()["run_id"]
        time.sleep(0.5)
        resp = client.get(f"/api/mine-logics/status/{run_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == run_id
        assert data["status"] in {"pending", "running", "completed", "failed", "stopped"}
        assert "progress" in data

    def test_status_nonexistent_returns_404(self):
        resp = client.get("/api/mine-logics/status/nonexistent")
        assert resp.status_code == 404


class TestMineLogicsResults:
    def test_results_pending(self):
        start_resp = client.post("/api/mine-logics/start", json={
            "source_libs": ["alpha101"],
            "max_per_lib": 1,
            "workers": 1,
        })
        run_id = start_resp.json()["run_id"]
        time.sleep(0.1)
        resp = client.get(f"/api/mine-logics/results/{run_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == run_id

    def test_results_completed(self):
        start_resp = client.post("/api/mine-logics/start", json={
            "source_libs": ["alpha101"],
            "max_per_lib": 1,
            "workers": 1,
        })
        run_id = start_resp.json()["run_id"]
        for _ in range(30):
            time.sleep(0.3)
            status_resp = client.get(f"/api/mine-logics/status/{run_id}")
            if status_resp.json()["status"] in {"completed", "failed"}:
                break
        resp = client.get(f"/api/mine-logics/results/{run_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in {"completed", "failed", "stopped"}

    def test_results_nonexistent_returns_404(self):
        resp = client.get("/api/mine-logics/results/nonexistent")
        assert resp.status_code == 404


class TestMineLogicsStop:
    def test_stop_returns_200(self):
        start_resp = client.post("/api/mine-logics/start", json={
            "source_libs": ["alpha101"],
            "max_per_lib": 1,
            "workers": 1,
        })
        run_id = start_resp.json()["run_id"]
        time.sleep(0.1)
        resp = client.post(f"/api/mine-logics/stop/{run_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert "stopped" in data

    def test_stop_nonexistent_returns_404(self):
        resp = client.post("/api/mine-logics/stop/nonexistent")
        assert resp.status_code == 404


class TestMineLogicsHistory:
    def test_history_returns_200(self):
        resp = client.get("/api/mine-logics/history")
        assert resp.status_code == 200
        data = resp.json()
        assert "runs" in data
        assert "total" in data
        assert isinstance(data["runs"], list)

    def test_history_includes_completed_run(self):
        start_resp = client.post("/api/mine-logics/start", json={
            "source_libs": ["alpha101"],
            "max_per_lib": 1,
            "workers": 1,
        })
        run_id = start_resp.json()["run_id"]
        for _ in range(30):
            time.sleep(0.3)
            status_resp = client.get(f"/api/mine-logics/status/{run_id}")
            if status_resp.json()["status"] in {"completed", "failed"}:
                break
        resp = client.get("/api/mine-logics/history")
        data = resp.json()
        assert any(r["run_id"] == run_id for r in data["runs"])
