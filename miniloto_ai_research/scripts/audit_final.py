"""Step 41: independent re-implementation / re-check of key results using
only the Python standard library (no pandas/numpy), to catch bugs that a
single-implementation pipeline could hide."""
import csv
import json
import math
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def main():
    checks = []

    def factorial(n):
        r = 1
        for i in range(2, n + 1):
            r *= i
        return r

    c315_manual = factorial(31) // (factorial(5) * factorial(26))
    checks.append(("C(31,5) manual factorial", c315_manual, 169911, c315_manual == 169911))

    with open(DATA / "miniloto_1_1404_clean.csv") as f:
        rows = list(csv.DictReader(f))
    checks.append(("row_count (stdlib csv)", len(rows), 1404, len(rows) == 1404))

    draw_nos = sorted(int(r["draw_no"]) for r in rows)
    checks.append(("draw_no unique", len(set(draw_nos)) == len(draw_nos), True, len(set(draw_nos)) == len(draw_nos)))
    checks.append(("draw_no range", (draw_nos[0], draw_nos[-1]), (1, 1404), (draw_nos[0], draw_nos[-1]) == (1, 1404)))

    total_main = 0
    all_vals = []
    main_counts = {i: 0 for i in range(1, 32)}
    for r in rows:
        nums = [int(r["n1"]), int(r["n2"]), int(r["n3"]), int(r["n4"]), int(r["n5"])]
        total_main += len(nums)
        all_vals.extend(nums)
        for x in nums:
            main_counts[x] += 1
    checks.append(("total_main_numbers", total_main, 7020, total_main == 7020))
    checks.append(("main_number_value_range", (min(all_vals), max(all_vals)), (1, 31), (min(all_vals), max(all_vals)) == (1, 31)))

    # cross-check against results/number_statistics.csv (pandas-generated)
    with open(RESULTS / "number_statistics.csv") as f:
        ns_rows = {int(r["number"]): int(r["main_count"]) for r in csv.DictReader(f)}
    mismatches = [n for n in range(1, 32) if ns_rows.get(n) != main_counts[n]]
    checks.append(("main_count matches pandas-generated number_statistics.csv for all 31 numbers", len(mismatches) == 0, True, len(mismatches) == 0))

    # cross-check combination count file
    with open(DATA / "all_169911_combinations.csv") as f:
        n_combos = sum(1 for _ in f) - 1
    checks.append(("all_169911_combinations.csv row count", n_combos, 169911, n_combos == 169911))

    # cross-check final_10.csv: 10 tickets, each 5 distinct numbers in 1..31
    with open(RESULTS / "final_10.csv") as f:
        final_rows = list(csv.DictReader(f))
    ok_final = len(final_rows) == 10
    for r in final_rows:
        nums = [int(r["n1"]), int(r["n2"]), int(r["n3"]), int(r["n4"]), int(r["n5"])]
        if len(set(nums)) != 5 or min(nums) < 1 or max(nums) > 31:
            ok_final = False
    checks.append(("final_10.csv structurally valid (10 tickets, 5 distinct numbers 1-31 each)", ok_final, True, ok_final))

    print("=== Independent Recompute / Final Audit (Step 41) ===")
    all_pass = True
    for name, obs, exp, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'} | {name}: observed={obs}, expected={exp}")
        all_pass = all_pass and ok
    print(f"\nOVERALL INDEPENDENT RECOMPUTE: {'PASS' if all_pass else 'FAIL'}")

    with open(RESULTS / "independent_recompute_audit.json", "w") as f:
        json.dump([{"check": n, "observed": str(o), "expected": str(e), "pass": ok} for n, o, e, ok in checks], f, indent=2)


if __name__ == "__main__":
    main()
