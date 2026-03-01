from __future__ import annotations

from langgraph.graph import StateGraph, END
from graph.state import WorkforceState
from graph.nodes import (
    ingest_hr_dataset_node,
    build_employee_skill_graph_node,
    identify_skill_gaps_node,
    compute_concentration_risk_node,
    simulate_attrition_node,
    compute_risk_score_node,
    recommend_hiring_reskilling_node,
    generate_board_ready_report_node,
    export_report_pdf_node,
    optional_push_to_neo4j_node,
    graph_rag_qa_node,
)


def build_workflow():
    wf = StateGraph(WorkforceState)

    wf.add_node("ingest_data", ingest_hr_dataset_node)
    wf.add_node("build_graph", build_employee_skill_graph_node)
    wf.add_node("analyze_gaps", identify_skill_gaps_node)
    wf.add_node("analyze_concentration", compute_concentration_risk_node)
    wf.add_node("simulate_attrition", simulate_attrition_node)
    wf.add_node("compute_risk", compute_risk_score_node)
    wf.add_node("make_recommendations", recommend_hiring_reskilling_node)
    wf.add_node("make_report", generate_board_ready_report_node)
    wf.add_node("export_pdf", export_report_pdf_node)

    # ✅ Always include Neo4j push
    wf.add_node("push_neo4j", optional_push_to_neo4j_node)

    wf.add_node("qa_graph_rag", graph_rag_qa_node)

    wf.set_entry_point("ingest_data")

    wf.add_edge("ingest_data", "build_graph")
    wf.add_edge("build_graph", "analyze_gaps")
    wf.add_edge("analyze_gaps", "analyze_concentration")
    wf.add_edge("analyze_concentration", "simulate_attrition")
    wf.add_edge("simulate_attrition", "compute_risk")
    wf.add_edge("compute_risk", "make_recommendations")
    wf.add_edge("make_recommendations", "make_report")
    wf.add_edge("make_report", "export_pdf")

    # ✅ FORCE push after PDF
    wf.add_edge("export_pdf", "push_neo4j")
    wf.add_edge("push_neo4j", "qa_graph_rag")

    wf.add_edge("qa_graph_rag", END)
    return wf.compile()