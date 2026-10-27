"""Compile the cortex phenomenon reports into Markdown tables with pass/fail verdicts.

Reads reports/cortex/phase{N}-{condition}.json, where condition is one of
untrained, newborn-sample, newborn-heldout and adult. Every verdict applies the
criterion written in the protocol (docs/protocols/) before the measurement.
"""

import argparse
import json
from pathlib import Path

CONDITIONS = [
    ("untrained", "Untrained"),
    ("newborn-sample", "Newborn, 2 min of the sample clip"),
    ("newborn-heldout", "Newborn, 2 min of a held-out UCF101 scene"),
    ("adult", "Adult"),
]


def load(folder: Path, phase: int) -> dict:
    out = {}
    for key, _ in CONDITIONS:
        path = folder / f"phase{phase}-{key}.json"
        if path.exists():
            out[key] = json.loads(path.read_text(encoding="utf-8"))
    return out


def fmt(value, digits=2) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def verdict(ok: bool | None) -> str:
    return "–" if ok is None else ("pass" if ok else "**fail**")


def orientation(reports: dict) -> list[str]:
    base = reports.get("untrained", {}).get("orientation", {}).get("median_osi")
    rows = ["| Condition | Median OSI | Cells with OSI > 0.3 | Protocol 1 |", "|---|---:|---:|---|"]
    for key, label in CONDITIONS:
        r = reports.get(key, {}).get("orientation")
        if not r:
            continue
        ok = (
            None
            if key == "untrained" or base is None
            else (r["median_osi"] >= base + 0.2 and r["fraction_selective"] >= 0.5)
        )
        rows.append(
            f"| {label} | {fmt(r['median_osi'])} | {r['fraction_selective']:.0%} | {verdict(ok)} |"
        )
    return rows


def oddball(reports: dict) -> list[str]:
    base = reports.get("untrained", {}).get("oddball", {})
    rows = [
        "| Condition | iRS (repetition suppression) | iPE (prediction error) | iMM (mismatch) | Protocol 2 |",
        "|---|---:|---:|---:|---|",
    ]
    for key, label in CONDITIONS:
        r = reports.get(key, {}).get("oddball")
        if not r:
            continue
        ok = (
            None
            if key == "untrained" or not base
            else (
                r["iMM"] > 0 and r["iRS"] > 0 and r["iMM"] > base["iMM"] and r["iRS"] > base["iRS"]
            )
        )
        rows.append(
            f"| {label} | {r['iRS']:+.2f} | {r['iPE']:+.2f} | {r['iMM']:+.2f} | {verdict(ok)} |"
        )
    return rows


def direction(reports: dict) -> list[str]:
    base = reports.get("untrained", {}).get("direction", {}).get("fraction_selective")
    rows = ["| Condition | Median DSI | Cells with DSI > 0.3 | Protocol 3 |", "|---|---:|---:|---|"]
    for key, label in CONDITIONS:
        r = reports.get(key, {}).get("direction")
        if not r:
            continue
        ok = (
            None
            if key == "untrained" or base is None
            else (r["fraction_selective"] >= 0.2 and r["fraction_selective"] > base)
        )
        rows.append(
            f"| {label} | {fmt(r['median_dsi'])} | {r['fraction_selective']:.0%} | {verdict(ok)} |"
        )
    return rows


def feedback_table(reports: dict, name: str, protocol: int, threshold: float) -> list[str]:
    rows = [
        f"| Condition | Median index with feedback | Without feedback | Difference | Protocol {protocol} |",
        "|---|---:|---:|---:|---|",
    ]
    for key, label in CONDITIONS:
        r = reports.get(key, {}).get(name)
        if not r:
            continue
        on, off = r["with_feedback"]["median"], r["without_feedback"]["median"]
        ok = (on - off >= threshold) if key == "adult" else None
        rows.append(f"| {label} | {fmt(on)} | {fmt(off)} | {on - off:+.2f} | {verdict(ok)} |")
    return rows


def remapping(reports: dict) -> list[str]:
    rows = [
        "| Condition | Remapping ratio (below 1: the efference copy helps) | Protocol 6 |",
        "|---|---:|---|",
    ]
    for key, label in CONDITIONS:
        r = reports.get(key, {}).get("remapping")
        if not r:
            continue
        ok = (r["remapping_ratio"] < 0.95) if key == "adult" else None
        rows.append(f"| {label} | {fmt(r['remapping_ratio'], 3)} | {verdict(ok)} |")
    return rows


def orienting(reports: dict) -> list[str]:
    control = (
        reports.get("untrained", {})
        .get("orienting", {})
        .get("frozen", {})
        .get("last", {})
        .get("toward_onsets")
    )
    rows = [
        "| Condition | Toward onsets, first 200 frames | Toward onsets, last 300 frames | Protocol 7 |",
        "|---|---:|---:|---|",
    ]
    if control is not None:
        frozen = reports["untrained"]["orienting"]["frozen"]
        rows.append(
            f"| Control: untrained, not learning | {fmt(frozen['first']['toward_onsets'])} | {fmt(control)} | – |"
        )
    for key, label in CONDITIONS:
        r = reports.get(key, {}).get("orienting", {}).get("learning")
        if not r:
            continue
        last = r["last"]["toward_onsets"]
        ok = (
            None if control is None or last is None or key == "untrained" else last >= control + 0.1
        )
        if key == "untrained":
            label = "Untrained, learning (a newborn)"
            ok = None if control is None or last is None else last >= control + 0.1
        rows.append(
            f"| {label} | {fmt(r['first']['toward_onsets'])} | {fmt(last)} | {verdict(ok)} |"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=Path("reports/cortex"))
    args = parser.parse_args()
    phase1, phase2, phase3 = (load(args.reports, phase) for phase in (1, 2, 3))
    sections = [
        ("Phase 1, protocol 1: orientation tuning", orientation(phase1)),
        ("Phase 1, protocol 2: oddball adaptation", oddball(phase1)),
        ("Phase 2, protocol 3: direction selectivity", direction(phase2)),
        ("Phase 2, protocol 4: surround suppression", feedback_table(phase2, "size", 4, 0.1)),
        ("Phase 2, protocol 5: end-stopping", feedback_table(phase2, "endstopping", 5, 1e-9)),
        ("Phase 3, protocol 6: predictive remapping", remapping(phase3)),
        ("Phase 3, protocol 7: habituation of orienting", orienting(phase3)),
    ]
    for title, rows in sections:
        print(f"#### {title}\n")
        print("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
