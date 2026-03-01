from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.encoders import jsonable_encoder
import json

# Ensure root import works
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graph.workflow import build_workflow
from tools.neo4j_tool import Neo4jTool
from tools.run_store import load_latest, save_run

app = FastAPI(title="Enterprise Talent Intelligence System")


# =====================================================
# Robust helpers (avoid 500 errors)
# =====================================================

def _html_page(title: str, body: str) -> str:
    return f"""
    <html>
    <head>
      <meta charset="utf-8"/>
      <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
      <style>
        body {{
          background:#0f172a;
          color:white;
          font-family:Arial;
          margin:0;
        }}
        .topbar {{
          padding:16px 20px;
          background:#111827;
          border-bottom:1px solid #1f2937;
          display:flex;
          gap:14px;
          align-items:center;
          flex-wrap:wrap;
        }}
        .brand {{
          font-weight:700;
          letter-spacing:0.3px;
        }}
        .nav a {{
          color:#93c5fd;
          text-decoration:none;
          margin-right:12px;
          font-weight:600;
          white-space:nowrap;
        }}
        .wrap {{ padding:22px; }}
        .card {{
          background:#1e293b;
          border:1px solid #334155;
          border-radius:12px;
          padding:16px;
          margin:12px 0;
        }}
        .kpis {{
          display:flex;
          gap:12px;
          flex-wrap:wrap;
        }}
        .kpi {{
          background:#111827;
          border:1px solid #334155;
          border-radius:12px;
          padding:14px;
          min-width:180px;
          flex: 1;
        }}
        .kpi .n {{
          font-size:24px;
          font-weight:800;
        }}
        table {{
          width:100%;
          border-collapse:collapse;
          margin-top:8px;
          font-size:14px;
        }}
        th, td {{
          border-bottom:1px solid #334155;
          padding:8px;
          text-align:left;
          vertical-align:top;
        }}
        th {{ color:#cbd5e1; }}
        .muted {{ color:#9ca3af; }}
        .btn {{
          display:inline-block;
          padding:10px 14px;
          background:#2563eb;
          color:white;
          text-decoration:none;
          border-radius:10px;
          font-weight:700;
        }}
        .btn2 {{
          display:inline-block;
          padding:10px 14px;
          background:#334155;
          color:white;
          text-decoration:none;
          border-radius:10px;
          font-weight:700;
          border:1px solid #475569;
        }}
        pre {{
          white-space:pre-wrap;
          word-wrap:break-word;
          background:#0b1220;
          border:1px solid #334155;
          border-radius:12px;
          padding:14px;
          overflow:auto;
        }}
        .grid2 {{
          display:grid;
          grid-template-columns: 1fr 1fr;
          gap:12px;
        }}
        @media(max-width: 980px) {{
          .grid2 {{ grid-template-columns: 1fr; }}
        }}
        code {{ color:#e5e7eb; }}
        .legend {{
          display:flex;
          gap:10px;
          flex-wrap:wrap;
          margin-top:10px;
        }}
        .chip {{
          display:inline-flex;
          align-items:center;
          gap:8px;
          background:#0b1220;
          border:1px solid #334155;
          border-radius:999px;
          padding:6px 10px;
          font-size:13px;
          color:#e5e7eb;
        }}
        .dot {{
          width:12px;
          height:12px;
          border-radius:50%;
          display:inline-block;
        }}
      </style>
      <title>{title}</title>
    </head>
    <body>
      <div class="topbar">
        <div class="brand">Enterprise Talent Intelligence System</div>
        <div class="nav">
          <a href="/dashboard">Dashboard</a>
          <a href="/graph_view">Graph</a>
          <a href="/graph_clustered">Clustered</a>
          <a href="/graph_board">Board Mode</a>
          <a href="/insights">Insights</a>
          <a href="/recommendations">Recommendations</a>
          <a href="/simulation">Simulation</a>
          <a href="/report">Report</a>
          <a href="/qa">GraphRAG Q&A</a>
          <a href="/docs">Docs</a>
        </div>
      </div>
      <div class="wrap">
        {body}
      </div>
    </body>
    </html>
    """


def _latest_payload() -> Optional[Dict[str, Any]]:
    wrapper = load_latest()
    if not wrapper:
        return None
    payload = wrapper.get("payload")
    return payload if isinstance(payload, dict) else None


def _try_parse_json_dict(val: Any) -> Dict[str, Any]:
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        s = val.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                parsed = json.loads(s)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
    return {}


def _count_table_like(val: Any, key: str) -> int:
    d = _try_parse_json_dict(val)
    col = d.get(key)
    if isinstance(col, dict):
        return len(col.keys())
    return 0


def _kpi_row(label: str, value: Any) -> str:
    return f'<div class="kpi"><div class="muted">{label}</div><div class="n">{value}</div></div>'


def _html_table(headers: List[str], rows: List[List[Any]]) -> str:
    th = "".join([f"<th>{h}</th>" for h in headers])
    trs = []
    for r in rows:
        tds = "".join([f"<td>{'' if v is None else v}</td>" for v in r])
        trs.append(f"<tr>{tds}</tr>")
    return f"<table><thead><tr>{th}</tr></thead><tbody>{''.join(trs)}</tbody></table>"


# =====================================================
# Risk heatmap helpers (B)
# =====================================================

def _risk_color(level: str) -> str:
    level = (level or "").upper()
    if level == "CRITICAL":
        return "#ef4444"  # red
    if level == "HIGH":
        return "#f97316"  # orange
    if level == "MEDIUM":
        return "#eab308"  # yellow
    if level == "LOW":
        return "#22c55e"  # green
    return "#3b82f6"      # default blue


def _skill_risk_map_from_latest() -> Dict[str, str]:
    payload = _latest_payload()
    if not payload:
        return {}
    risks = payload.get("concentration_risks") or []
    if not isinstance(risks, list):
        return {}
    out: Dict[str, str] = {}
    for r in risks:
        if not isinstance(r, dict):
            continue
        skill = r.get("skill")
        level = r.get("risk_level", "")
        if skill:
            out[str(skill)] = str(level)
    return out


# =====================================================
# Core routes
# =====================================================

@app.get("/")
def root():
    return {
        "message": "Enterprise Talent Intelligence System Running",
        "docs": "/docs",
        "dashboard": "/dashboard",
        "graph_view": "/graph_view",
        "graph_clustered": "/graph_clustered",
        "graph_board": "/graph_board",
        "insights": "/insights",
        "recommendations": "/recommendations",
        "simulation": "/simulation",
        "report": "/report",
        "qa": "/qa",
        "health": "/health",
    }


@app.get("/health")
def health():
    tool = Neo4jTool()
    status = tool.connect()
    tool.close()
    latest = load_latest()
    return {
        "ok": True,
        "neo4j_enabled": status.connected,
        "neo4j_message": status.message,
        "latest_run_available": bool(latest),
    }


# =====================================================
# ✅ POST /run (Multipart Form) — appears in /docs
#   FIXED: returns JSON-safe response only
# =====================================================

@app.post("/run")
async def run_workflow(
    employees: UploadFile = File(...),
    projects: UploadFile = File(...),
    performance: UploadFile = File(None),
    question: str = Form("Which critical skills are concentrated in only one team?"),
    simulate_role: str = Form("Senior Architect"),
    simulate_count: int = Form(2),
):
    files = {
        "employees.csv": await employees.read(),
        "projects.csv": await projects.read(),
    }
    if performance:
        files["performance.csv"] = await performance.read()

    wf = build_workflow()
    result = wf.invoke(
        {
            "uploaded_files": files,
            "question": question,
            "simulate_role": simulate_role,
            "simulate_count": simulate_count,
            "report_pdf_path": str(PROJECT_ROOT / "workforce_strategy_report.pdf"),
        }
    )

    # Save full run output for UI pages
    run_id = save_run(result)

    # Return a small, always-JSON-safe response
    safe_response = {
        "ok": True,
        "run_id": run_id,
        "question": question,
        "answer": str(result.get("answer", "")),
        "report_pdf_path": str(result.get("report_pdf_path", "")),
        "risk_score": result.get("risk_score", {}),
        "counts": {
            "skill_gaps": len(result.get("skill_gaps", []) or []),
            "concentration_risks": len(result.get("concentration_risks", []) or []),
            "department_needs": len(result.get("department_needs", []) or []),
        },
        "next": {
            "dashboard": "/dashboard",
            "graph_view": "/graph_view",
            "graph_clustered": "/graph_clustered",
            "graph_board": "/graph_board",
            "insights": "/insights",
            "recommendations": "/recommendations",
            "simulation": "/simulation",
            "report": "/report",
            "qa": "/qa",
        },
    }
    return JSONResponse(content=jsonable_encoder(safe_response))


# =====================================================
# Dashboard
# =====================================================

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    payload = _latest_payload()
    if not payload:
        body = """
        <div class="card">
          <h2>No run found</h2>
          <p class="muted">
            Open <code>/docs</code> → run <b>POST /run</b> once, then refresh.
          </p>
        </div>
        """
        return HTMLResponse(_html_page("Dashboard", body))

    employees = payload.get("employees")
    projects = payload.get("projects")
    performance = payload.get("performance")

    risks = payload.get("concentration_risks") or []
    gaps = payload.get("skill_gaps") or []
    risk_score = payload.get("risk_score") or {}

    if not isinstance(risks, list): risks = []
    if not isinstance(gaps, list): gaps = []
    if not isinstance(risk_score, dict): risk_score = {}

    emp_count = _count_table_like(employees, "employee_id")
    proj_count = _count_table_like(projects, "project_id")
    perf_count = _count_table_like(performance, "employee_id")

    body = f"""
      <div class="card">
        <h2>Executive Overview</h2>
        <div class="kpis">
          {_kpi_row("Employees", emp_count)}
          {_kpi_row("Projects", proj_count)}
          {_kpi_row("Performance Reviews", perf_count)}
          {_kpi_row("Skill Gaps (projects)", len(gaps))}
          {_kpi_row("Concentration Risks", len(risks))}
          {_kpi_row("Workforce Risk Level", risk_score.get("level", "NA"))}
        </div>
      </div>

      <div class="grid2">
        <div class="card">
          <h3>What it does</h3>
          <ul>
            <li>Maps Employees, Skills, Projects, Departments, Performance Reviews</li>
            <li>Identifies skill gaps and single-point-of-failure skills</li>
            <li>Recommends hiring & reskilling plan</li>
            <li>Generates board-ready workforce strategy report</li>
            <li>GraphRAG answers workforce risk questions</li>
          </ul>
        </div>

        <div class="card">
          <h3>Graph Views</h3>
          <p><a class="btn" href="/graph_view">Standard Graph</a></p>
          <p><a class="btn" href="/graph_clustered">Clustered Graph</a></p>
          <p><a class="btn" href="/graph_board">Board Mode</a></p>
        </div>
      </div>
    """
    return HTMLResponse(_html_page("Dashboard", body))


# =====================================================
# Graph View (standard)
# =====================================================

@app.get("/graph_view", response_class=HTMLResponse)
def graph_view():
    tool = Neo4jTool()
    status = tool.connect()

    if not status.connected:
        tool.close()
        body = f"""
          <div class="card">
            <h2>Neo4j not connected</h2>
            <p class="muted">{status.message}</p>
            <p class="muted">Start Neo4j and rerun <b>POST /run</b>.</p>
          </div>
        """
        return HTMLResponse(_html_page("Graph View", body))

    records = tool.run_query("""
        MATCH (a:Entity)-[r:REL]->(b:Entity)
        RETURN a,r,b
        LIMIT 550
    """)
    tool.close()

    skill_risk = _skill_risk_map_from_latest()

    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []

    for rec in records:
        a = rec["a"]
        b = rec["b"]
        r = rec["r"]

        for node in (a, b):
            nid = str(node.id)
            ntype = node.get("type", "Entity")
            nname = node.get("name", nid)

            if nid not in nodes:
                if ntype == "Skill":
                    level = skill_risk.get(str(nname), "")
                    nodes[nid] = {
                        "id": nid,
                        "label": str(nname),
                        "group": "Skill",
                        "color": _risk_color(level),
                        "title": f"Skill: {nname} | Risk: {level or 'NA'}",
                    }
                else:
                    nodes[nid] = {
                        "id": nid,
                        "label": str(nname),
                        "group": str(ntype),
                        "title": f"{ntype}: {nname}",
                    }

        edges.append({
            "from": str(a.id),
            "to": str(b.id),
            "label": r.get("type", ""),
            "arrows": "to",
        })

    nodes_json = json.dumps(list(nodes.values()), ensure_ascii=False).replace("</", "<\\/")
    edges_json = json.dumps(edges, ensure_ascii=False).replace("</", "<\\/")

    legend = f"""
      <div class="legend">
        <span class="chip"><span class="dot" style="background:#8b5cf6"></span>Department</span>
        <span class="chip"><span class="dot" style="background:#22c55e"></span>Employee</span>
        <span class="chip"><span class="dot" style="background:#f59e0b"></span>Project</span>
        <span class="chip"><span class="dot" style="background:#3b82f6"></span>Skill (default)</span>
        <span class="chip"><span class="dot" style="background:#22c55e"></span>LOW</span>
        <span class="chip"><span class="dot" style="background:#eab308"></span>MEDIUM</span>
        <span class="chip"><span class="dot" style="background:#f97316"></span>HIGH</span>
        <span class="chip"><span class="dot" style="background:#ef4444"></span>CRITICAL</span>
      </div>
    """

    body = f"""
      <div class="card">
        <h2>Workforce Graph (Standard)</h2>
        <p class="muted">Skills are colored by concentration risk level when available.</p>
        <div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:10px;">
          <a class="btn" href="/graph_clustered">Clustered View</a>
          <a class="btn2" href="/graph_board">Board Mode</a>
        </div>
        {legend}
        <div id="network" style="height:80vh;border:1px solid #334155;border-radius:12px;margin-top:14px;"></div>
      </div>

      <script>
        var nodes = new vis.DataSet({nodes_json});
        var edges = new vis.DataSet({edges_json});

        var options = {{
          layout: {{
            hierarchical: {{
              enabled: true,
              direction: "LR",
              levelSeparation: 240,
              nodeSpacing: 180,
              treeSpacing: 260
            }}
          }},
          physics: false,
          nodes: {{
            shape: "box",
            margin: 10,
            font: {{ color: "white", size: 14 }}
          }},
          groups: {{
            Department: {{ color: "#8b5cf6" }},
            Employee: {{ color: "#22c55e" }},
            Project: {{ color: "#f59e0b" }},
            Skill: {{ color: "#3b82f6" }}
          }},
          edges: {{
            arrows: {{ to: {{ enabled: true }} }},
            smooth: false,
            color: "#94a3b8",
            font: {{ color: "#cbd5e1", size: 12 }}
          }},
          interaction: {{
            hover: true,
            zoomView: true,
            dragView: true
          }}
        }};

        new vis.Network(document.getElementById("network"), {{nodes:nodes, edges:edges}}, options);
      </script>
    """
    return HTMLResponse(_html_page("Graph View", body))


# =====================================================
# Clustered Graph View (C)
# =====================================================

@app.get("/graph_clustered", response_class=HTMLResponse)
def graph_clustered():
    tool = Neo4jTool()
    status = tool.connect()

    if not status.connected:
        tool.close()
        body = f"""
          <div class="card">
            <h2>Neo4j not connected</h2>
            <p class="muted">{status.message}</p>
          </div>
        """
        return HTMLResponse(_html_page("Clustered Graph", body))

    records = tool.run_query("""
        MATCH (a:Entity)-[r:REL]->(b:Entity)
        RETURN a,r,b
        LIMIT 650
    """)
    tool.close()

    skill_risk = _skill_risk_map_from_latest()

    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []

    for rec in records:
        a = rec["a"]
        b = rec["b"]
        r = rec["r"]

        for node in (a, b):
            nid = str(node.id)
            ntype = node.get("type", "Entity")
            nname = node.get("name", nid)

            if nid not in nodes:
                if ntype == "Skill":
                    level = skill_risk.get(str(nname), "")
                    nodes[nid] = {
                        "id": nid,
                        "label": str(nname),
                        "group": "Skill",
                        "color": _risk_color(level),
                        "title": f"Skill: {nname} | Risk: {level or 'NA'}",
                    }
                else:
                    nodes[nid] = {
                        "id": nid,
                        "label": str(nname),
                        "group": str(ntype),
                        "title": f"{ntype}: {nname}",
                    }

        edges.append({
            "from": str(a.id),
            "to": str(b.id),
            "label": r.get("type", ""),
            "arrows": "to",
        })

    nodes_json = json.dumps(list(nodes.values()), ensure_ascii=False).replace("</", "<\\/")
    edges_json = json.dumps(edges, ensure_ascii=False).replace("</", "<\\/")

    body = f"""
      <div class="card">
        <h2>Clustered Graph (Executive Readable)</h2>
        <p class="muted">Left → Right layout for better readability. Skills are risk-colored.</p>
        <div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:10px;">
          <a class="btn" href="/graph_view">Standard View</a>
          <a class="btn2" href="/graph_board">Board Mode</a>
        </div>
        <div id="network" style="height:80vh;border:1px solid #334155;border-radius:12px;margin-top:14px;"></div>
      </div>

      <script>
        var nodes = new vis.DataSet({nodes_json});
        var edges = new vis.DataSet({edges_json});

        var options = {{
          layout: {{
            hierarchical: {{
              enabled: true,
              direction: "LR",
              levelSeparation: 280,
              nodeSpacing: 190,
              treeSpacing: 300
            }}
          }},
          physics: false,
          nodes: {{
            shape: "box",
            margin: 10,
            font: {{ color: "white", size: 14 }}
          }},
          groups: {{
            Department: {{ color: "#8b5cf6" }},
            Employee: {{ color: "#22c55e" }},
            Project: {{ color: "#f59e0b" }},
            Skill: {{ color: "#3b82f6" }}
          }},
          edges: {{
            arrows: {{ to: {{ enabled: true }} }},
            smooth: false,
            color: "#94a3b8",
            font: {{ color: "#cbd5e1", size: 12 }}
          }},
          interaction: {{
            hover: true,
            zoomView: true,
            dragView: true
          }}
        }};

        new vis.Network(document.getElementById("network"), {{nodes:nodes, edges:edges}}, options);
      </script>
    """
    return HTMLResponse(_html_page("Clustered Graph", body))


# =====================================================
# Board Presentation Mode (D)
# =====================================================

@app.get("/graph_board", response_class=HTMLResponse)
def graph_board():
    payload = _latest_payload() or {}
    risk_score = payload.get("risk_score") or {}
    if not isinstance(risk_score, dict):
        risk_score = {}

    tool = Neo4jTool()
    status = tool.connect()
    if not status.connected:
        tool.close()
        body = f"""
          <div class="card">
            <h2>Neo4j not connected</h2>
            <p class="muted">{status.message}</p>
          </div>
        """
        return HTMLResponse(_html_page("Board Mode", body))

    records = tool.run_query("""
        MATCH (a:Entity)-[r:REL]->(b:Entity)
        WHERE r.type IN ['BELONGS_TO','WORKS_ON','HAS_SKILL','REQUIRES','NEEDS']
        RETURN a,r,b
        LIMIT 450
    """)
    tool.close()

    skill_risk = _skill_risk_map_from_latest()

    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []

    for rec in records:
        a = rec["a"]
        b = rec["b"]
        r = rec["r"]

        for node in (a, b):
            nid = str(node.id)
            ntype = node.get("type", "Entity")
            nname = node.get("name", nid)

            if nid not in nodes:
                if ntype == "Skill":
                    level = skill_risk.get(str(nname), "")
                    nodes[nid] = {
                        "id": nid,
                        "label": str(nname),
                        "group": "Skill",
                        "color": _risk_color(level),
                        "title": f"Skill: {nname} | Risk: {level or 'NA'}",
                    }
                else:
                    nodes[nid] = {
                        "id": nid,
                        "label": str(nname),
                        "group": str(ntype),
                        "title": f"{ntype}: {nname}",
                    }

        edges.append({
            "from": str(a.id),
            "to": str(b.id),
            "label": r.get("type", ""),
            "arrows": "to",
        })

    nodes_json = json.dumps(list(nodes.values()), ensure_ascii=False).replace("</", "<\\/")
    edges_json = json.dumps(edges, ensure_ascii=False).replace("</", "<\\/")

    legend = f"""
      <div class="legend">
        <span class="chip"><span class="dot" style="background:#22c55e"></span>LOW</span>
        <span class="chip"><span class="dot" style="background:#eab308"></span>MEDIUM</span>
        <span class="chip"><span class="dot" style="background:#f97316"></span>HIGH</span>
        <span class="chip"><span class="dot" style="background:#ef4444"></span>CRITICAL</span>
      </div>
    """

    body = f"""
      <div class="card">
        <h2>Board Presentation Mode</h2>
        <div class="kpis">
          {_kpi_row("Risk Score", risk_score.get("score","NA"))}
          {_kpi_row("Risk Level", risk_score.get("level","NA"))}
          {_kpi_row("Resilience Index", risk_score.get("resilience_index","NA"))}
        </div>
        <p class="muted">Shows only key relations: BELONGS_TO, WORKS_ON, HAS_SKILL, REQUIRES, NEEDS. Skills are risk-colored.</p>
        <div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:10px;">
          <a class="btn" href="/graph_clustered">Clustered View</a>
          <a class="btn2" href="/graph_view">Standard View</a>
        </div>
        {legend}
        <div id="network" style="height:80vh;border:1px solid #334155;border-radius:12px;margin-top:14px;"></div>
      </div>

      <script>
        var nodes = new vis.DataSet({nodes_json});
        var edges = new vis.DataSet({edges_json});

        var options = {{
          layout: {{
            hierarchical: {{
              enabled: true,
              direction: "LR",
              levelSeparation: 300,
              nodeSpacing: 210,
              treeSpacing: 320
            }}
          }},
          physics: false,
          nodes: {{
            shape: "box",
            margin: 10,
            font: {{ color: "white", size: 14 }}
          }},
          groups: {{
            Department: {{ color: "#8b5cf6" }},
            Employee: {{ color: "#22c55e" }},
            Project: {{ color: "#f59e0b" }},
            Skill: {{ color: "#3b82f6" }}
          }},
          edges: {{
            arrows: {{ to: {{ enabled: true }} }},
            smooth: false,
            color: "#94a3b8",
            font: {{ color: "#cbd5e1", size: 12 }}
          }},
          interaction: {{
            hover: true,
            zoomView: true,
            dragView: true
          }}
        }};

        new vis.Network(document.getElementById("network"), {{nodes:nodes, edges:edges}}, options);
      </script>
    """
    return HTMLResponse(_html_page("Board Mode", body))


# =====================================================
# Insights
# =====================================================

@app.get("/insights", response_class=HTMLResponse)
def insights():
    payload = _latest_payload()
    if not payload:
        return HTMLResponse(_html_page("Insights", "<div class='card'><h2>No run found</h2><p class='muted'>Run POST /run in /docs and refresh.</p></div>"))

    risks = payload.get("concentration_risks") or []
    dept_needs = payload.get("department_needs") or []

    if not isinstance(risks, list): risks = []
    if not isinstance(dept_needs, list): dept_needs = []

    risks_rows = [[r.get("skill"), r.get("employee_count"), json.dumps(r.get("by_department", {})), r.get("risk_level"), r.get("single_team_only")] for r in risks if isinstance(r, dict)]

    needs_rows: List[List[Any]] = []
    for d in dept_needs:
        if not isinstance(d, dict):
            continue
        dept = d.get("department")
        for need in d.get("needs", []):
            if not isinstance(need, dict):
                continue
            needs_rows.append([dept, need.get("skill"), need.get("demand")])

    body = f"""
      <div class="card">
        <h2>Insights</h2>
        <p class="muted">Concentration risk + department needs.</p>
      </div>

      <div class="card">
        <h3>Concentration Risks</h3>
        {_html_table(["Skill","Holders","By Department","Risk","Single Team"], risks_rows[:100])}
      </div>

      <div class="card">
        <h3>Department Needs</h3>
        {_html_table(["Department","Skill","Demand"], needs_rows[:200])}
      </div>
    """
    return HTMLResponse(_html_page("Insights", body))


# =====================================================
# Recommendations
# =====================================================

@app.get("/recommendations", response_class=HTMLResponse)
def recommendations():
    payload = _latest_payload()
    if not payload:
        return HTMLResponse(_html_page("Recommendations", "<div class='card'><h2>No run found</h2><p class='muted'>Run POST /run in /docs and refresh.</p></div>"))

    recs = payload.get("recommendations") or {}
    if not isinstance(recs, dict):
        recs = {}

    hiring = recs.get("hiring_plan") or []
    reskill = recs.get("reskilling_plan") or []
    executive = recs.get("executive_summary") or ""
    llm_meta = recs.get("llm_meta") or {}

    if not isinstance(hiring, list): hiring = []
    if not isinstance(reskill, list): reskill = []
    if not isinstance(llm_meta, dict): llm_meta = {}

    hiring_rows = [[x.get("skill"), x.get("priority_score"), x.get("reason")] for x in hiring if isinstance(x, dict)]
    reskill_rows = [[x.get("skill"), x.get("priority_score"), x.get("reason")] for x in reskill if isinstance(x, dict)]

    body = f"""
      <div class="card">
        <h2>Recommendations</h2>
        <div class="kpis">
          {_kpi_row("LLM Provider", llm_meta.get("provider","NA"))}
          {_kpi_row("LLM Model", llm_meta.get("model","NA"))}
          {_kpi_row("Reachable", llm_meta.get("reachable","NA"))}
        </div>
      </div>

      <div class="card">
        <h3>Hiring Plan</h3>
        {_html_table(["Skill","Priority","Reason"], hiring_rows[:60])}
      </div>

      <div class="card">
        <h3>Reskilling Plan</h3>
        {_html_table(["Skill","Priority","Reason"], reskill_rows[:60])}
      </div>

      <div class="card">
        <h3>Executive Summary</h3>
        <pre>{executive}</pre>
      </div>
    """
    return HTMLResponse(_html_page("Recommendations", body))


# =====================================================
# Simulation
# =====================================================

@app.get("/simulation", response_class=HTMLResponse)
def simulation():
    payload = _latest_payload()
    if not payload:
        return HTMLResponse(_html_page("Simulation", "<div class='card'><h2>No run found</h2><p class='muted'>Run POST /run in /docs and refresh.</p></div>"))

    sim = payload.get("simulation") or {}
    if not isinstance(sim, dict):
        sim = {}

    removed = sim.get("removed_employees") or []
    impacted = sim.get("impacted_projects") or []
    if not isinstance(removed, list): removed = []
    if not isinstance(impacted, list): impacted = []

    body = f"""
      <div class="card">
        <h2>Attrition Simulation</h2>
        <p class="muted">{payload.get("question","")}</p>
      </div>

      <div class="grid2">
        <div class="card">
          <h3>Removed Employees</h3>
          {_html_table(["Employee"], [[x] for x in removed]) if removed else "<p class='muted'>No removed employees in output.</p>"}
        </div>
        <div class="card">
          <h3>Impacted Projects</h3>
          {_html_table(["Project"], [[x] for x in impacted]) if impacted else "<p class='muted'>No impacted projects detected.</p>"}
        </div>
      </div>

      <div class="card">
        <h3>LLM Answer</h3>
        <pre>{payload.get("answer","")}</pre>
      </div>
    """
    return HTMLResponse(_html_page("Simulation", body))


# =====================================================
# Report
# =====================================================

@app.get("/report", response_class=HTMLResponse)
def report():
    payload = _latest_payload()
    if not payload:
        return HTMLResponse(_html_page("Report", "<div class='card'><h2>No run found</h2><p class='muted'>Run POST /run in /docs and refresh.</p></div>"))

    md = payload.get("report_markdown") or ""
    pdf_path = str(payload.get("report_pdf_path") or "")
    p = Path(pdf_path) if pdf_path else None

    if p and p.exists():
        download_btn = '<p><a class="btn" href="/download_latest_pdf">Download PDF</a></p>'
    else:
        download_btn = f"<p class='muted'>PDF not found at: <code>{pdf_path}</code> (Install reportlab + rerun)</p>"

    body = f"""
      <div class="card">
        <h2>Board-Ready Report</h2>
        {download_btn}
      </div>
      <div class="card">
        <h3>Markdown</h3>
        <pre>{md}</pre>
      </div>
    """
    return HTMLResponse(_html_page("Report", body))


@app.get("/download_latest_pdf")
def download_latest_pdf():
    payload = _latest_payload()
    if not payload:
        return HTMLResponse(_html_page("Download PDF", "<div class='card'><h2>No run found</h2></div>"))

    pdf_path = str(payload.get("report_pdf_path") or "")
    p = Path(pdf_path) if pdf_path else None
    if not p or not p.exists():
        return HTMLResponse(_html_page("Download PDF", f"<div class='card'><h2>PDF not found</h2><p class='muted'>{pdf_path}</p></div>"))

    return FileResponse(path=str(p), filename=p.name, media_type="application/pdf")


# =====================================================
# GraphRAG Q&A View
# =====================================================

@app.get("/qa", response_class=HTMLResponse)
def qa():
    payload = _latest_payload()
    if not payload:
        return HTMLResponse(_html_page("GraphRAG Q&A", "<div class='card'><h2>No run found</h2><p class='muted'>Run POST /run in /docs and refresh.</p></div>"))

    q = payload.get("question") or ""
    ans = payload.get("answer") or ""
    ctx = payload.get("rag_context") or ""

    body = f"""
      <div class="card">
        <h2>GraphRAG Q&A</h2>
        <p class="muted">Answer + graph context used.</p>
      </div>

      <div class="card">
        <h3>Question</h3>
        <pre>{q}</pre>
      </div>

      <div class="card">
        <h3>Answer</h3>
        <pre>{ans}</pre>
      </div>

      <div class="card">
        <h3>Graph Context</h3>
        <pre>{ctx}</pre>
      </div>

      <div class="card">
        <h3>Try these questions</h3>
        <ul>
          <li>Which critical skills are concentrated in only one team?</li>
          <li>What happens if 2 senior architects leave?</li>
          <li>Which departments have the highest single-point skill risks?</li>
          <li>What should we hire for next quarter to reduce risk?</li>
          <li>Which projects are most vulnerable due to skill concentration?</li>
        </ul>
      </div>
    """
    return HTMLResponse(_html_page("GraphRAG Q&A", body))