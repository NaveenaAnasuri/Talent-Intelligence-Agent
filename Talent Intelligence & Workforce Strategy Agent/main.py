from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graph.workflow import build_workflow
from tools.run_store import save_run  # ✅ NEW


def main():
    parser = argparse.ArgumentParser(description="Enterprise Talent Intelligence System (V3)")
    parser.add_argument("--employees", required=True, help="Path to employees.csv")
    parser.add_argument("--projects", required=True, help="Path to projects.csv")
    parser.add_argument("--performance", default="", help="Optional path to performance.csv")
    parser.add_argument("--question", default="Which critical skills are concentrated in only one team?")
    parser.add_argument("--simulate_role", default="Senior Architect")
    parser.add_argument("--simulate_count", type=int, default=2)
    args = parser.parse_args()

    files = {
        "employees.csv": Path(args.employees).read_bytes(),
        "projects.csv": Path(args.projects).read_bytes(),
    }
    if args.performance:
        files["performance.csv"] = Path(args.performance).read_bytes()

    wf = build_workflow()
    out = wf.invoke(
        {
            "uploaded_files": files,
            "question": args.question,
            "simulate_role": args.simulate_role,
            "simulate_count": args.simulate_count,
            "report_pdf_path": str(PROJECT_ROOT / "workforce_strategy_report.pdf"),
        }
    )

    # ✅ NEW: Save output for the dashboard pages (insights/recommendations/report/qa/etc.)
    run_id = save_run(out)
    out["run_id"] = run_id

    print("\n✅ Run saved. run_id =", run_id)
    print("📌 Latest run saved to: runs/latest.json")

    print("\n=== BOARD REPORT (Markdown) ===\n")
    print(out.get("report_markdown", ""))

    print("\n=== GRAPH RAG ANSWER ===\n")
    print(out.get("answer", ""))

    print("\nPDF saved to:", out.get("report_pdf_path"))


if __name__ == "__main__":
    main()