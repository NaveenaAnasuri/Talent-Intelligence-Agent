from __future__ import annotations

from typing import Any, Dict, List, Tuple
import networkx as nx


def build_rag_context(
    g: nx.MultiDiGraph,
    question: str,
    max_nodes: int = 40,
    max_edges: int = 80,
) -> str:
    """
    Lightweight GraphRAG context builder:
    - Select relevant nodes by keyword match on node id and 'name'
    - Expand to neighbor edges
    """
    q = (question or "").lower().strip()
    if not q:
        return "No question provided."

    # Score nodes
    scored: List[Tuple[float, str]] = []
    for n, attrs in g.nodes(data=True):
        text = f"{n} {attrs.get('name','')} {attrs.get('type','')}".lower()
        score = 0.0
        for token in q.split():
            if token and token in text:
                score += 1.0
        if score > 0:
            scored.append((score, n))

    scored.sort(reverse=True)
    seed_nodes = [n for _, n in scored[: max_nodes // 2]]  # half seeds

    # Expand neighbors
    selected = set(seed_nodes)
    for n in list(seed_nodes):
        for u, v, attrs in list(g.in_edges(n, data=True))[:10]:
            selected.add(u)
            selected.add(v)
        for u, v, attrs in list(g.out_edges(n, data=True))[:10]:
            selected.add(u)
            selected.add(v)
        if len(selected) >= max_nodes:
            break

    # Build text context
    lines: List[str] = []
    lines.append("GRAPH CONTEXT (nodes):")
    for n in list(selected)[:max_nodes]:
        a = g.nodes[n]
        lines.append(f"- {n} | type={a.get('type')} | name={a.get('name','')}".strip())

    lines.append("\nGRAPH CONTEXT (edges):")
    edge_count = 0
    for u, v, attrs in g.edges(data=True):
        if u in selected and v in selected:
            lines.append(f"- {u} -[{attrs.get('type','REL')}]-> {v} | props={_short_props(attrs)}")
            edge_count += 1
            if edge_count >= max_edges:
                break

    return "\n".join(lines)


def _short_props(d: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        if k == "type":
            continue
        if isinstance(v, (str, int, float, bool)) and len(str(v)) <= 60:
            out[k] = v
    return out