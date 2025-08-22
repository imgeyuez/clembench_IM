# analyze_referencegame.py
#
# Usage option A (inline data):
#   - Paste your `data = [...]` list above this script and run it.
#
# Usage option B (from JSON on disk):
#   - Save your dataset to dataset.json (a list of dicts) and run:
#       python analyze_referencegame.py --json dataset.json
#
# The script prints a summary and saves charts under ./figures

import json
import os
from collections import Counter, defaultdict
from typing import List, Dict, Any

import matplotlib.pyplot as plt


# ---------- Helper aggregation ----------
ROLE_MAP = {
    "comp": "comprehension",
    "gen": "generation",
}

def normalize_role(raw_role: str) -> str:
    if raw_role in ROLE_MAP:
        return ROLE_MAP[raw_role]
    # Fallback to raw if unknown
    return str(raw_role) if raw_role else "unknown"

def normalize_status(raw_status: str) -> str:
    if not raw_status:
        return "unknown"
    s = raw_status.strip().lower()

    return s

def outcome_from_reward(reward) -> str:
    # Only meaningful for finished games
    try:
        r = int(reward)
    except (TypeError, ValueError):
        return "other"
    if r == 1:
        return "success"
    if r == -1:
        return "failure"

def aggregate(rows: List[Dict[str, Any]]):
    # Totals by role
    totals_by_role = Counter()
    # Status counts by role
    status_by_role = defaultdict(Counter)
    # Finished outcomes by role (success/failure/other)
    finished_outcomes_by_role = defaultdict(Counter)

    for row in rows:
        role = normalize_role(row.get("role"))
        status = normalize_status(row.get("status"))
        reward = row.get("reward", None)

        totals_by_role[role] += 1
        status_by_role[role][status] += 1

        if status == "finished":
            finished_outcomes_by_role[role][outcome_from_reward(reward)] += 1

    return totals_by_role, status_by_role, finished_outcomes_by_role

# ---------- Visualization helpers (one figure per chart) ----------
def ensure_outdir(outdir="figures"):
    os.makedirs(outdir, exist_ok=True)
    return outdir

def pie_role_distribution(totals_by_role: Counter, outdir="figures"):
    labels = list(totals_by_role.keys())
    sizes = [totals_by_role[k] for k in labels]

    plt.figure()
    plt.title("Dataset by Task Type")
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    plt.axis("equal")
    path = os.path.join(outdir, "role_distribution_pie.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    return path

def bar_status_by_role(status_by_role: Dict[str, Counter], outdir="figures"):
    roles = sorted(status_by_role.keys())
    statuses = ["finished", "aborted"]

    # Prepare grouped bars
    x = range(len(roles))
    width = 0.8 / len(statuses)

    plt.figure()
    plt.title("Status by Task Type")
    plt.ylabel("Count")
    for i, st in enumerate(statuses):
        heights = [status_by_role[r].get(st, 0) for r in roles]
        # Shift bars for grouped effect
        xs = [xi + (i - (len(statuses)-1)/2) * width for xi in x]
        plt.bar(xs, heights, width=width, label=st)

    plt.xticks(list(x), roles, rotation=0)
    plt.legend()
    path = os.path.join(outdir, "status_by_role_bar.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    return path

def bar_finished_outcomes_by_role(finished_outcomes_by_role: Dict[str, Counter], outdir="figures"):
    roles = sorted(finished_outcomes_by_role.keys())
    outcomes = ["success", "failure"]

    x = range(len(roles))
    width = 0.8 / len(outcomes)

    plt.figure()
    plt.title("Finished Games: Outcomes by Task Type")
    plt.ylabel("Count")
    for i, oc in enumerate(outcomes):
        heights = [finished_outcomes_by_role[r].get(oc, 0) for r in roles]
        xs = [xi + (i - (len(outcomes)-1)/2) * width for xi in x]
        plt.bar(xs, heights, width=width, label=oc)

    plt.xticks(list(x), roles, rotation=0)
    plt.legend()
    path = os.path.join(outdir, "finished_outcomes_by_role_bar.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    return path

# ---------- Pretty printing ----------
def print_summary(totals_by_role, status_by_role, finished_outcomes_by_role):
    all_roles = sorted(set(totals_by_role.keys()) |
                       set(status_by_role.keys()) |
                       set(finished_outcomes_by_role.keys()))

    print("\n=== SUMMARY ===")
    grand_total = sum(totals_by_role.values())
    print(f"Total items: {grand_total}")
    print("\nBy task type:")
    for r in all_roles:
        print(f"  - {r}: {totals_by_role.get(r, 0)}")

    print("\nStatus by task type:")
    for r in all_roles:
        st = status_by_role.get(r, {})
        print(f"  - {r}: finished={st.get('finished',0)}, aborted={st.get('aborted',0)}")

    print("\nFinished outcomes by task type (reward):")
    for r in all_roles:
        oc = finished_outcomes_by_role.get(r, {})
        print(f"  - {r}: success={oc.get('success',0)}, failure={oc.get('failure',0)}")
    print("===============")

# ---------- Main ----------
def main(rows: List[Dict[str, Any]]):
    totals_by_role, status_by_role, finished_outcomes_by_role = aggregate(rows)
    print_summary(totals_by_role, status_by_role, finished_outcomes_by_role)

    outdir = ensure_outdir("figures")
    paths = []
    paths.append(pie_role_distribution(totals_by_role, outdir))
    paths.append(bar_status_by_role(status_by_role, outdir))
    paths.append(bar_finished_outcomes_by_role(finished_outcomes_by_role, outdir))

    print("\nSaved figures:")
    for p in paths:
        print(f" - {p}")

if __name__ == "__main__":

    DATAPATH = ''
    with open(r'C:\Users\imgey\Desktop\MASTERS\MASTER_POTSDAM\SoSe25\IM\codespace\referencegame_data.json', 'r') as f:
        rows = json.load(f)

    main(rows)
