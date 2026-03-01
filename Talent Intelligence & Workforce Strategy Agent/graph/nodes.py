from __future__ import annotations

import io
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import pandas as pd
import networkx as nx

from tools.api_connectors import LLMClient
from tools.neo4j_tool import Neo4jTool
from analysis.graph_rag import build_rag_context


def _split_list(cell: Any, seps: Tuple[str, ...] = (",", ";", "|")) -> List[str]:
    if cell is None:
        return []
    s = str(cell).strip()
    if not s or s.lower() == "nan":
        return []
    for sep in seps:
        s = s.replace(sep, ",")
    return [p.strip() for p in s.split(",") if p.strip()]


def ingest_hr_dataset_node(state: Dict[str, Any]) -> Dict[str, Any]:
    files: Dict[str, bytes] = state.get("uploaded_files", {})

    if "employees.csv" not in files or "projects.csv" not in files:
        raise ValueError("Missing required files: employees.csv and projects.csv")

    employees = pd.read_csv(io.BytesIO(files["employees.csv"]))
    projects = pd.read_csv(io.BytesIO(files["projects.csv"]))

    performance = pd.DataFrame()
    if "performance.csv" in files:
        performance = pd.read_csv(io.BytesIO(files["performance.csv"]))

    # Normalize column names lightly
    if "name" in employees.columns and "employee_name" not in employees.columns:
        employees = employees.rename(columns={"name": "employee_name"})

    return {"employees": employees, "projects": projects, "performance": performance}


def build_employee_skill_graph_node(state: Dict[str, Any]) -> Dict[str, Any]:
    employees: pd.DataFrame = state["employees"]
    projects: pd.DataFrame = state["projects"]
    performance: pd.DataFrame = state["performance"]

    g = nx.MultiDiGraph()

    # Departments
    if "department" in employees.columns:
        for dept in sorted(set(employees["department"].dropna().astype(str))):
            if dept.strip():
                g.add_node(f"dept:{dept}", type="Department", name=dept)

    # Projects
    for _, r in projects.iterrows():
        pid = str(r.get("project_id", "")).strip()
        if not pid:
            continue
        pname = str(r.get("project_name", pid)).strip()
        pdept = str(r.get("project_department", r.get("department", ""))).strip()
        g.add_node(f"proj:{pid}", type="Project", project_id=pid, name=pname, department=pdept)
        if pdept:
            g.add_node(f"dept:{pdept}", type="Department", name=pdept)

    # Employees + BELONGS_TO + HAS_SKILL + WORKS_ON
    for _, r in employees.iterrows():
        eid = str(r.get("employee_id", "")).strip()
        if not eid:
            continue
        name = str(r.get("employee_name", eid)).strip()
        dept = str(r.get("department", "")).strip()
        role = str(r.get("role", "")).strip()
        skills = _split_list(r.get("skills", ""))

        g.add_node(f"emp:{eid}", type="Employee", employee_id=eid, name=name, department=dept, role=role)
        if dept:
            g.add_node(f"dept:{dept}", type="Department", name=dept)
            g.add_edge(f"emp:{eid}", f"dept:{dept}", type="BELONGS_TO")

        for sk in skills:
            g.add_node(f"skill:{sk}", type="Skill", name=sk)
            g.add_edge(f"emp:{eid}", f"skill:{sk}", type="HAS_SKILL")

        # Employee -> WORKS_ON -> Project (expects project_ids column)
        if "project_ids" in employees.columns:
            for pid in _split_list(r.get("project_ids", "")):
                if pid:
                    g.add_node(f"proj:{pid}", type="Project", project_id=pid, name=pid)
                    g.add_edge(f"emp:{eid}", f"proj:{pid}", type="WORKS_ON")

    # Project -> REQUIRES -> Skill
    for _, r in projects.iterrows():
        pid = str(r.get("project_id", "")).strip()
        if not pid:
            continue
        req = _split_list(r.get("required_skills", ""))
        for sk in req:
            g.add_node(f"skill:{sk}", type="Skill", name=sk)
            g.add_edge(f"proj:{pid}", f"skill:{sk}", type="REQUIRES")

    # Department -> NEEDS -> Skill (derived from dept projects required skills)
    dept_skill_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for _, r in projects.iterrows():
        pid = str(r.get("project_id", "")).strip()
        if not pid:
            continue
        pdept = str(r.get("project_department", r.get("department", ""))).strip()
        if not pdept:
            continue
        for sk in _split_list(r.get("required_skills", "")):
            dept_skill_counts[pdept][sk] += 1

    department_needs = []
    for dept, skmap in dept_skill_counts.items():
        g.add_node(f"dept:{dept}", type="Department", name=dept)
        for sk, cnt in skmap.items():
            g.add_node(f"skill:{sk}", type="Skill", name=sk)
            g.add_edge(f"dept:{dept}", f"skill:{sk}", type="NEEDS", demand=int(cnt))
        department_needs.append({"department": dept, "needs": [{"skill": sk, "demand": int(cnt)} for sk, cnt in skmap.items()]})

    # Attach performance reviews as node properties (if present)
    if not performance.empty and "employee_id" in performance.columns:
        perf_map = {}
        for _, r in performance.iterrows():
            eid = str(r.get("employee_id", "")).strip()
            if not eid:
                continue
            perf_map[eid] = {
                "performance_rating": r.get("performance_rating", r.get("rating", "")),
                "performance_notes": r.get("performance_notes", r.get("notes", "")),
            }
        for eid, props in perf_map.items():
            n = f"emp:{eid}"
            if n in g.nodes:
                for k, v in props.items():
                    g.nodes[n][k] = v

    return {"graph": g, "department_needs": department_needs}


def identify_skill_gaps_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Skill gaps per project:
    required skills - skills available among employees who WORKS_ON that project.
    Fallback: if no WORKS_ON edges, use department employees.
    """
    g: nx.MultiDiGraph = state["graph"]
    skill_gaps: List[Dict[str, Any]] = []

    # Map project -> workers
    proj_workers: Dict[str, List[str]] = defaultdict(list)
    for u, v, data in g.edges(data=True):
        if data.get("type") == "WORKS_ON" and u.startswith("emp:") and v.startswith("proj:"):
            proj_workers[v].append(u)

    # Map emp -> skills
    emp_skills: Dict[str, set] = defaultdict(set)
    for u, v, data in g.edges(data=True):
        if data.get("type") == "HAS_SKILL" and u.startswith("emp:") and v.startswith("skill:"):
            emp_skills[u].add(v.split("skill:", 1)[1])

    # Map dept -> employees
    dept_emps: Dict[str, List[str]] = defaultdict(list)
    for u, v, data in g.edges(data=True):
        if data.get("type") == "BELONGS_TO" and u.startswith("emp:") and v.startswith("dept:"):
            dept_emps[v].append(u)

    for n, attrs in g.nodes(data=True):
        if attrs.get("type") != "Project":
            continue

        dept = str(attrs.get("department", "")).strip()
        required = set()
        for _, v, ed in g.out_edges(n, data=True):
            if ed.get("type") == "REQUIRES" and str(v).startswith("skill:"):
                required.add(str(v).split("skill:", 1)[1])

        workers = proj_workers.get(n, [])
        available = set()
        if workers:
            for e in workers:
                available |= emp_skills.get(e, set())
        else:
            # fallback dept level
            dept_node = f"dept:{dept}" if dept else ""
            for e in dept_emps.get(dept_node, []):
                available |= emp_skills.get(e, set())

        missing = sorted(list(required - available))
        if missing:
            skill_gaps.append({
                "project_id": attrs.get("project_id", n.replace("proj:", "")),
                "project_name": attrs.get("name", ""),
                "department": dept,
                "missing_skills": missing,
            })

    return {"skill_gaps": skill_gaps}


def compute_concentration_risk_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Concentration risk: skills held by <=2 employees, and whether they are confined to a single department/team.
    """
    g: nx.MultiDiGraph = state["graph"]
    skill_to_emps: Dict[str, set] = defaultdict(set)
    emp_dept: Dict[str, str] = {}

    for n, a in g.nodes(data=True):
        if a.get("type") == "Employee":
            emp_dept[n] = str(a.get("department", "")).strip()

    for u, v, d in g.edges(data=True):
        if d.get("type") == "HAS_SKILL" and u.startswith("emp:") and v.startswith("skill:"):
            skill = v.split("skill:", 1)[1]
            skill_to_emps[skill].add(u)

    risks = []
    for sk, emps in skill_to_emps.items():
        if len(emps) <= 2:
            by_dept = defaultdict(int)
            for e in emps:
                by_dept[emp_dept.get(e, "")] += 1
            risks.append({
                "skill": sk,
                "employee_count": len(emps),
                "by_department": dict(by_dept),
                "risk_level": "HIGH" if len(emps) == 1 else "MEDIUM",
                "single_team_only": len([d for d in by_dept.keys() if d]) <= 1,
            })

    risks.sort(key=lambda x: (x["risk_level"] != "HIGH", x["employee_count"]))
    return {"concentration_risks": risks}


def simulate_attrition_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Scenario: remove N employees of a given role; recompute affected projects (no workers) and newly-missing skills.
    """
    role = (state.get("simulate_role") or "").strip()
    count = int(state.get("simulate_count") or 0)
    g: nx.MultiDiGraph = state["graph"]

    if not role or count <= 0:
        return {"simulation": {"removed_employees": [], "impacted_projects": [], "notes": "No simulation requested."}}

    # pick employees by role (deterministic order)
    emps = [n for n, a in g.nodes(data=True) if a.get("type") == "Employee" and str(a.get("role","")).strip() == role]
    to_remove = sorted(emps)[:count]

    g2 = g.copy()
    g2.remove_nodes_from(to_remove)

    # impacted projects: no WORKS_ON edges left
    impacted = []
    for n, a in g2.nodes(data=True):
        if a.get("type") == "Project":
            has_worker = any(ed.get("type") == "WORKS_ON" for _, _, ed in g2.in_edges(n, data=True))
            if not has_worker:
                impacted.append({"project_id": a.get("project_id", n.replace("proj:", "")), "project_name": a.get("name", "")})

    return {"simulation": {"removed_employees": to_remove, "impacted_projects": impacted}}


def compute_risk_score_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simple enterprise-friendly risk score:
    - gaps weighted 2
    - high concentration weighted 3, medium weighted 1
    - impacted projects weighted 2
    """
    gaps = state.get("skill_gaps", [])
    risks = state.get("concentration_risks", [])
    sim = state.get("simulation", {})

    score = 0
    score += 2 * len(gaps)
    for r in risks:
        score += 3 if r.get("risk_level") == "HIGH" else 1
    score += 2 * len(sim.get("impacted_projects", []))

    if score >= 18:
        level = "CRITICAL"
    elif score >= 10:
        level = "HIGH"
    elif score >= 4:
        level = "MEDIUM"
    else:
        level = "LOW"

    resilience_index = max(0, 100 - score * 4)

    return {"risk_score": {"score": score, "level": level, "resilience_index": resilience_index}}


def recommend_hiring_reskilling_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic recommendations + optional LLM narrative.
    """
    gaps = state.get("skill_gaps", [])
    risks = state.get("concentration_risks", [])

    # Rank skills by frequency in gaps + concentration
    priority = defaultdict(int)
    for g in gaps:
        for sk in g.get("missing_skills", []):
            priority[sk] += 2
    for r in risks:
        sk = r.get("skill")
        if sk:
            priority[sk] += 3 if r.get("risk_level") == "HIGH" else 1

    ranked = sorted(priority.items(), key=lambda x: x[1], reverse=True)

    hiring = []
    reskill = []
    for sk, sc in ranked[:20]:
        high = any(r.get("skill") == sk and r.get("risk_level") == "HIGH" for r in risks)
        if high or sc >= 4:
            hiring.append({"skill": sk, "priority_score": sc, "reason": "Critical gap / high concentration risk"})
        else:
            reskill.append({"skill": sk, "priority_score": sc, "reason": "Reskill internal staff (targeted upskilling)"})

    llm = LLMClient()
    system = (
        "You are a Talent Intelligence & Workforce Strategy assistant for HR and enterprise workforce planning. "
        "Write concise executive recommendations. Use bullet points."
    )
    user = (
        f"Risk score: {state.get('risk_score')}\n"
        f"Top gaps: {gaps[:8]}\n"
        f"Top concentration risks: {risks[:8]}\n"
        f"Simulation: {state.get('simulation')}\n\n"
        "Provide an executive action plan: hiring priorities, reskilling plan, and risk mitigation steps."
    )
    llm_resp = llm.chat(system=system, user=user)

    return {
        "recommendations": {
            "priority_skills": [{"skill": sk, "score": sc} for sk, sc in ranked],
            "hiring_plan": hiring,
            "reskilling_plan": reskill,
            "executive_summary": llm_resp.text,
            "llm_meta": llm_resp.meta,
        }
    }


def generate_board_ready_report_node(state: Dict[str, Any]) -> Dict[str, Any]:
    risk = state.get("risk_score", {})
    gaps = state.get("skill_gaps", [])
    risks = state.get("concentration_risks", [])
    sim = state.get("simulation", {})
    recs = state.get("recommendations", {})

    md = []
    md.append("# Board-Ready Workforce Strategy Report\n")
    md.append("## Executive Summary")
    md.append(f"- Workforce Risk Level: **{risk.get('level','')}** (score={risk.get('score','')}, resilience={risk.get('resilience_index','')} )")
    md.append(f"- Projects with Skill Gaps: **{len(gaps)}**")
    md.append(f"- Concentration Risks: **{len(risks)}**")
    md.append(f"- Attrition Simulation Impacted Projects: **{len(sim.get('impacted_projects', []))}**\n")

    md.append("## Top Skill Gaps (sample)")
    for g in gaps[:10]:
        md.append(f"- **{g.get('project_name','')}** ({g.get('department','')}): missing → {', '.join(g.get('missing_skills', [])[:10])}")

    md.append("\n## Top Concentration Risks (sample)")
    for r in risks[:10]:
        md.append(f"- **{r.get('skill')}**: holders={r.get('employee_count')} risk={r.get('risk_level')} single_team_only={r.get('single_team_only')}")

    md.append("\n## Attrition Simulation")
    md.append(f"- Removed employees: {sim.get('removed_employees', [])}")
    md.append(f"- Impacted projects: {[p.get('project_name','') for p in sim.get('impacted_projects', [])]}\n")

    md.append("## Recommendations")
    md.append("### Hiring priorities (Top 10)")
    for h in recs.get("hiring_plan", [])[:10]:
        md.append(f"- Hire for **{h['skill']}** (score={h['priority_score']}): {h['reason']}")

    md.append("\n### Reskilling priorities (Top 10)")
    for r in recs.get("reskilling_plan", [])[:10]:
        md.append(f"- Reskill into **{r['skill']}** (score={r['priority_score']}): {r['reason']}")

    md.append("\n### Executive narrative (LLM)")
    md.append(recs.get("executive_summary", ""))

    return {"report_markdown": "\n".join(md)}


def export_report_pdf_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates a simple PDF from markdown text.
    """
    report_md: str = state.get("report_markdown", "")
    out_path = state.get("report_pdf_path") or "workforce_strategy_report.pdf"
    if not report_md.strip():
        return {"report_pdf_path": None}

    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(out_path, pagesize=letter)
    width, height = letter

    y = height - 50
    for line in report_md.splitlines():
        if y < 50:
            c.showPage()
            y = height - 50
        c.drawString(40, y, line[:140])
        y -= 14

    c.save()
    return {"report_pdf_path": out_path}


def optional_push_to_neo4j_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Always returns neo4j_status so you can see success/failure in /run response.
    If ENABLE_NEO4J=false or neo4j driver missing, it will return a clear message.
    """
    tool = Neo4jTool()
    status = tool.connect()

    if not status.connected:
        tool.close()
        return {"neo4j_status": status.message}

    g: nx.MultiDiGraph = state["graph"]

    nodes = []
    for n, a in g.nodes(data=True):
        ntype = a.get("type", "Entity")
        props = dict(a)
        props["name"] = props.get("name", n)
        nodes.append((str(n), str(ntype), props))

    edges = []
    for u, v, a in g.edges(data=True):
        rel = a.get("type", "REL")
        edges.append((str(u), str(rel), str(v), dict(a)))

    msg = tool.upsert_graph(nodes=nodes, edges=edges)
    tool.close()
    return {"neo4j_status": msg}


def graph_rag_qa_node(state: Dict[str, Any]) -> Dict[str, Any]:
    question = (state.get("question") or "").strip()
    if not question:
        return {"rag_context": "", "answer": ""}

    g: nx.MultiDiGraph = state["graph"]
    context = build_rag_context(g, question=question, max_nodes=40, max_edges=80)

    llm = LLMClient()
    system = (
        "You are a Workforce Strategy AI. Use the provided graph context only. "
        "Answer clearly, with risks and mitigation steps. Keep it executive."
    )
    user = f"{context}\n\nQuestion: {question}"
    resp = llm.chat(system=system, user=user)

    return {"rag_context": context, "answer": resp.text}