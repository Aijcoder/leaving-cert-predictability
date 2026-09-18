"""One entry point for the pipeline (§17 reproduction).

    python -m lcbank.run --from-phase 5      # labelling onwards
    python -m lcbank.run --phases 6,7        # just these phases
    python -m lcbank.run --list

Phase 2 onwards needs the exam PDFs in data_private/raw/; phases 5-6 need the LLM API.
resumable and safe to re-run: finished work is skipped.
"""
import argparse
import subprocess
import sys

PHASES = {
    2: ("Extract page text and images, quality report",
        ["python -m lcbank.extract.extract_pdfs", "python -m lcbank.extract.quality_report"]),
    3: ("Structure: parts, marks, rules trees, crops",
        ["python -m lcbank.structure.build_phy", "python -m lcbank.structure.build_maths"]),
    4: ("Codebooks from syllabus headings (frozen v1)",
        ["python -m lcbank.codebook.phy_outline", "python -m lcbank.codebook.build_phy",
         "python -m lcbank.codebook.pm_outline", "python -m lcbank.codebook.build_pm",
         "python -m lcbank.codebook.build_am2"]),
    5: ("LLM labelling and merge (needs LLM_* in .env)",
        ["python -m lcbank.label.label_parts --dataset PHY --items all --run final --version v4s",
         "python -m lcbank.label.label_parts --dataset PM --items all --run final --version v4s",
         "python -m lcbank.label.label_parts --dataset PMT --items all --run final --version v4s",
         "python -m lcbank.label.label_parts --dataset AM2 --items all --run final --version v4s",
         "python -m lcbank.label.merge_labels --run final --freeze"]),
    6: ("Label QA: automatic checks A1-A5",
        ["python -m lcbank.qa.auto_checks --run final"]),
    7: ("Topic x sitting tables and per-topic history",
        ["python -m lcbank.analysis.tables"]),
    8: ("Analysis: simulator controls, then the real run and figures",
        ["python -m lcbank.analysis.sim --runs 500 --perm 99 --workers 1 --shard 0 --shards 6",
         "python -m lcbank.analysis.run_analysis", "python -m lcbank.analysis.figures"]),
    9: ("Exports (public and full tiers)",
        ["python -m lcbank.export.build_exports"]),
    10: ("Final summary", ["python -m lcbank.report.final_summary"]),
}


def run_phase(n, dry_run=False):
    title, commands = PHASES[n]
    print(f"\n=== Phase {n}: {title}", flush=True)
    for cmd in commands:
        print(f"$ {cmd}", flush=True)
        if dry_run:
            continue
        result = subprocess.run(cmd.replace("python", sys.executable, 1), shell=True)
        if result.returncode != 0:
            raise SystemExit(f"Phase {n} stopped: `{cmd}` exited with {result.returncode}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-phase", type=int)
    ap.add_argument("--phases", help="comma-separated list, e.g. 6,7")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.list or not (a.from_phase or a.phases):
        for n, (title, cmds) in PHASES.items():
            print(f"{n:>2}  {title}")
            for c in cmds:
                print(f"      {c}")
        raise SystemExit(0)
    todo = [int(x) for x in a.phases.split(",")] if a.phases else [n for n in PHASES if n >= a.from_phase]
    for n in todo:
        run_phase(n, a.dry_run)
