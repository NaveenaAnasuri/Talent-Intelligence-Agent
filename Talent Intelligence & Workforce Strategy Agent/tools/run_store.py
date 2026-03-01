from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, List


RUNS_DIR = Path("runs")
LATEST_PATH = RUNS_DIR / "latest.json"


def ensure_runs_dir() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


def _make_json_safe(obj: Any) -> Any:
    """
    Convert non-JSON-serializable objects into JSON-safe representations.
    - bytes -> {"__type__":"bytes","size":N}
    - Path  -> str(path)
    - set/tuple -> list
    - fallback -> str(obj)
    """
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, bytes):
        return {"__type__": "bytes", "size": len(obj)}

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, (list, tuple, set)):
        return [_make_json_safe(x) for x in obj]

    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}

    try:
        return str(obj)
    except Exception:
        return repr(obj)


def _sanitize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove/compact large or non-serializable fields but keep metadata.
    - uploaded_files bytes removed (keeps filename + size)
    """
    payload = dict(payload)  # shallow copy

    if "uploaded_files" in payload and isinstance(payload["uploaded_files"], dict):
        compact = {}
        for fname, content in payload["uploaded_files"].items():
            if isinstance(content, bytes):
                compact[fname] = {"size": len(content)}
            else:
                compact[fname] = _make_json_safe(content)
        payload["uploaded_files"] = compact

    return _make_json_safe(payload)


def save_run(payload: Dict[str, Any]) -> str:
    """
    Saves a workflow run payload as runs/<run_id>.json and also updates runs/latest.json.
    Returns run_id.
    """
    ensure_runs_dir()
    run_id = uuid.uuid4().hex[:12]
    ts = datetime.now(timezone.utc).isoformat()

    try:
        safe_payload = _sanitize_payload(payload)
    except Exception:
        # fallback: store only important keys
        safe_payload = _sanitize_payload({
            "question": payload.get("question"),
            "answer": payload.get("answer"),
            "report_markdown": payload.get("report_markdown"),
            "report_pdf_path": payload.get("report_pdf_path"),
            "recommendations": payload.get("recommendations"),
            "simulation": payload.get("simulation"),
            "risk_score": payload.get("risk_score"),
            "skill_gaps": payload.get("skill_gaps"),
            "concentration_risks": payload.get("concentration_risks"),
            "department_needs": payload.get("department_needs"),
            "employees": payload.get("employees"),
            "projects": payload.get("projects"),
            "performance": payload.get("performance"),
            "rag_context": payload.get("rag_context"),
            "uploaded_files": payload.get("uploaded_files"),
        })

    wrapper = {
        "run_id": run_id,
        "saved_at_utc": ts,
        "payload": safe_payload,
    }

    run_path = RUNS_DIR / f"{run_id}.json"
    run_path.write_text(json.dumps(wrapper, indent=2, ensure_ascii=False), encoding="utf-8")
    LATEST_PATH.write_text(json.dumps(wrapper, indent=2, ensure_ascii=False), encoding="utf-8")
    return run_id


def load_latest() -> Optional[Dict[str, Any]]:
    if not LATEST_PATH.exists():
        return None
    try:
        return json.loads(LATEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_run(run_id: str) -> Optional[Dict[str, Any]]:
    run_path = RUNS_DIR / f"{run_id}.json"
    if not run_path.exists():
        return None
    try:
        return json.loads(run_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def list_runs(limit: int = 20) -> List[Dict[str, Any]]:
    ensure_runs_dir()
    files = sorted(RUNS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    out: List[Dict[str, Any]] = []
    for p in files[:limit]:
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return out