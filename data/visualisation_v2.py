# analyze_referencegame_with_ds.py
#
# Usage option A (inline data):
#   - Paste your `data = [...]` list above this script and run it.
#
# Usage option B (from JSON on disk):
#   - Save your dataset to dataset.json (a list of dicts) and run:
#       python analyze_referencegame_with_ds.py --json dataset.json
#
# The script prints a summary and saves charts under ./figures

import json
import os
from collections import Counter, defaultdict
from typing import List, Dict, Any

import matplotlib.pyplot as plt

# ---------------- Normalization helpers ----------------

ROLE_MAP = {"comp": "comprehension", "gen": "generation"}

def normalize_role(raw_role: str) -> str:
    if not raw_role:
        return "unknown"
    return ROLE_MAP.get(raw_role, str(raw_role))

def normalize_status(raw_status: str) -> str:
    if not raw_status:
        return "unknown"
    s = raw_status.strip().lower()
    if s in {"finished", "aborted"}:
        return s
    return "unknown"

def is_datashare(source: Any) -> bool:
    """Datasharing datapoints have `_DS` at the end of the last path segment."""
    if source is None:
        return False
    base = os.path.basename(str(source).rstrip("/"))
    return base.endswith("_DS")

def outcome_from_reward(reward) -> str:
    """Only used for finished games. We ignore any reward not {1,-1}."""
    try:
        r = int(reward)
    except (TypeError, ValueError):
        return "ignore"
    if r == 1:
        return "success"
    if r == -1:
        return "failure"
    return "ignore"

# ---------------- Aggregation ----------------

def aggregate(rows: List[Dict[str, Any]]):
    # Totals by (role, ds)
    totals_by_role_ds = Counter()                 # key: (role, ds_bool)
    # Status counts by (role, ds)
    status_by_role_ds = defaultdict(Counter)      # (role, ds) -> {finished, aborted, unknown}
    # Outcomes (finished only) by (role, ds)
    finished_outcomes_by_role_ds = defaultdict(Counter)  # (role, ds) -> {success, failure}

    for row in rows:
        role = normalize_role(row.get("role"))
        ds = is_datashare(row.get("source"))
        status = normalize_status(row.get("status"))
        reward = row.get("reward", None)

        key = (role, ds)
        totals_by_role_ds[key] += 1
        status_by_role_ds[key][status] += 1

        if status == "finished":
            oc = outcome_from_reward(reward)
            if oc in ("success", "failure"):
                finished_outcomes_by_role_ds[key][oc] += 1

    return totals_by_role_ds, status_by_role_ds, finished_outcomes_by_role_ds

# ---------------- I/O helpers ----------------

def ensure_outdir(outdir="figures"):
    os.makedirs(outdir, exist_ok=True)
    return outdir

# ---------------- Charts (one chart per figure) ----------------

def pie_role_ds_distribution(totals_by_role_ds: Counter, outdir="figures"):
    """
    One pie with four slices:
      gen no-DS (blue), gen DS (light blue),
      comp no-DS (salmon), comp DS (violet)
    """
    # Build labels & sizes in a fixed logical order for consistent colors
    combos = [
        ("generation", False, "Generation (no DS)", "blue"),
        ("generation", True,  "Generation (DS)", "#ADD8E6"),  # light blue
        ("comprehension", False, "Comprehension (no DS)", "salmon"),
        ("comprehension", True,  "Comprehension (DS)", "violet"),
    ]

    labels = []
    sizes = []
    colors = []
    for role, ds, label, color in combos:
        v = totals_by_role_ds.get((role, ds), 0)
        if v > 0:
            labels.append(label)
            sizes.append(v)
            colors.append(color)

    if not sizes:
        return None

    plt.figure()
    plt.title("Datapoints by Task Type and Data Sharing")
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors)
    plt.axis("equal")
    path = os.path.join(outdir, "role_ds_distribution_pie.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    return path

def bar_status_by_role_for_ds(status_by_role_ds: Dict, ds_flag: bool, outdir="figures"):
    """
    Grouped bars: status (finished/aborted) per role, filtered by ds_flag.
    Colors left to matplotlib defaults (simple).
    """
    roles = ["generation", "comprehension"]
    statuses = ["finished", "aborted"]

    # Keep only roles that exist for this ds_flag
    roles_present = [r for r in roles if sum(status_by_role_ds.get((r, ds_flag), {}).values()) > 0]
    if not roles_present:
        return None

    x = range(len(roles_present))
    width = 0.8 / len(statuses)

    plt.figure()
    title = "Status by Task Type (with DS)" if ds_flag else "Status by Task Type (no DS)"
    plt.title(title)
    plt.ylabel("Count")

    for i, st in enumerate(statuses):
        heights = [status_by_role_ds.get((r, ds_flag), {}).get(st, 0) for r in roles_present]
        xs = [xi + (i - (len(statuses)-1)/2) * width for xi in x]
        plt.bar(xs, heights, width=width, label=st)

    plt.xticks(list(x), roles_present)
    plt.legend()
    fname = "status_by_role_ds_bar.png" if ds_flag else "status_by_role_no_ds_bar.png"
    path = os.path.join(outdir, fname)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    return path

def bar_outcomes_by_role_for_ds(finished_outcomes_by_role_ds: Dict, ds_flag: bool, outdir="figures"):
    """
    Grouped bars: success vs failure per role, filtered by ds_flag.
    success = blue, failure = pink (as requested).
    """
    roles = ["generation", "comprehension"]
    roles_present = [r for r in roles if sum(finished_outcomes_by_role_ds.get((r, ds_flag), {}).values()) > 0]
    if not roles_present:
        return None

    x = range(len(roles_present))
    w = 0.4

    success_vals = [finished_outcomes_by_role_ds.get((r, ds_flag), {}).get("success", 0) for r in roles_present]
    failure_vals = [finished_outcomes_by_role_ds.get((r, ds_flag), {}).get("failure", 0) for r in roles_present]

    plt.figure()
    title = "Finished: Success vs Failure by Task Type (with DS)" if ds_flag else "Finished: Success vs Failure by Task Type (no DS)"
    plt.title(title)
    plt.ylabel("Count")

    # Two simple calls; colors explicit per your preference
    plt.bar([xi - w/2 for xi in x], success_vals, width=w, label="success", color="blue")
    plt.bar([xi + w/2 for xi in x], failure_vals, width=w, label="failure", color="pink")

    plt.xticks(list(x), roles_present)
    plt.legend()
    fname = "finished_outcomes_by_role_ds_bar.png" if ds_flag else "finished_outcomes_by_role_no_ds_bar.png"
    path = os.path.join(outdir, fname)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    return path

# ---------------- Summaries ----------------

def print_summary(totals_by_role_ds, status_by_role_ds, finished_outcomes_by_role_ds):
    def block_for(ds_flag: bool) -> str:
        return "WITH datasharing" if ds_flag else "NO datasharing"

    roles = ["generation", "comprehension"]
    ds_options = [False, True]

    print("\n=== SUMMARY (by role and datasharing) ===")
    grand_total = sum(totals_by_role_ds.values())
    print(f"Total items: {grand_total}")

    print("\nCounts by role & datasharing:")
    for ds in ds_options:
        print(f"  [{block_for(ds)}]")
        for r in roles:
            print(f"    - {r}: {totals_by_role_ds.get((r, ds), 0)}")

    print("\nStatus by role & datasharing:")
    for ds in ds_options:
        print(f"  [{block_for(ds)}]")
        for r in roles:
            st = status_by_role_ds.get((r, ds), {})
            print(f"    - {r}: finished={st.get('finished',0)}, aborted={st.get('aborted',0)}")

    print("\nFinished outcomes (reward) by role & datasharing:")
    for ds in ds_options:
        print(f"  [{block_for(ds)}]")
        for r in roles:
            oc = finished_outcomes_by_role_ds.get((r, ds), {})
            print(f"    - {r}: success={oc.get('success',0)}, failure={oc.get('failure',0)}")
    print("=========================================")

# ---------------- Main ----------------

def main(rows: List[Dict[str, Any]]):
    totals_by_role_ds, status_by_role_ds, finished_outcomes_by_role_ds = aggregate(rows)
    print_summary(totals_by_role_ds, status_by_role_ds, finished_outcomes_by_role_ds)

    outdir = ensure_outdir("figures")
    saved = []
    # (1) Combined pie with 4 slices (role × DS)
    saved.append(pie_role_ds_distribution(totals_by_role_ds, outdir))
    # (2) + (3) Status charts (no DS, with DS)
    saved.append(bar_status_by_role_for_ds(status_by_role_ds, ds_flag=False, outdir=outdir))
    saved.append(bar_status_by_role_for_ds(status_by_role_ds, ds_flag=True, outdir=outdir))
    # (4) + (5) Outcomes charts (no DS, with DS)
    saved.append(bar_outcomes_by_role_for_ds(finished_outcomes_by_role_ds, ds_flag=False, outdir=outdir))
    saved.append(bar_outcomes_by_role_for_ds(finished_outcomes_by_role_ds, ds_flag=True, outdir=outdir))

    print("\nSaved figures:")
    for p in saved:
        if p:
            print(f" - {p}")

if __name__ == "__main__":
    DATAPATH = r'C:\Users\imgey\Desktop\MASTERS\MASTER_POTSDAM\SoSe25\IM\codespace\referencegame_data_DS.json'
    with open(DATAPATH, 'r') as f:
        rows = json.load(f)

    main(rows)
