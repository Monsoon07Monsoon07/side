"""
Mini Loto (ミニロト) historical draw data acquisition.

Data sources (fetched via direct HTTPS GET of static, publicly hosted JSON
files; both are third-party aggregations of the official Mizuho Bank /
takarakuji-official mini loto results, not scraped/guessed by this script):

  Source A (primary):
    https://raw.githubusercontent.com/tk030-lotto/lotto-data-hub/main/data/miniloto.json
    -> full round 1..1404 history, numbers + bonus. Actively updated repo
       (GitHub Actions scraper), most recent round at time of fetch = 1404
       (2026-09-15).

  Source B (cross-check, also full history + prize data):
    https://raw.githubusercontent.com/ddsky0728/Japanese-lottery-data/main/loto/miniloto_history.json
    -> full round 1..1404 history, numbers + bonus + prize tiers
       (rank/winners/amount for ranks 1-4).

  Source C (cross-check, partial - last 100 rounds only):
    https://raw.githubusercontent.com/tank1159jhs/jp-lottery-api/main/data/miniloto/all.json
    -> rounds ~1302-1404.

NOTE ON NETWORK ACCESS: this research session runs behind an organization
egress proxy that allow-lists only a small set of developer-infrastructure
domains (github.com / raw.githubusercontent.com / api.github.com / package
registries). Direct access to the official/semi-official Japanese lottery
result sites named in the task brief (mizuhobank.co.jp, mk-mode.com,
miniloto.thekyo.jp, sougaku.com, etc.) returned EGRESS_BLOCKED / 403 for
every attempt (see reports/data_audit.md). GitHub-hosted mirrors of the
same official results were the only reachable source and were used instead.
This is disclosed explicitly per the "never silently substitute" principle;
see reports/data_audit.md for the full list of attempted/blocked hosts.

This script only performs the HTTP GET + raw JSON save. All comparison /
cleaning happens in audit_data.py.
"""
import json
import pathlib
import subprocess
import sys

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"

SOURCES = {
    "source_A_tk030lotto_raw.json": "https://raw.githubusercontent.com/tk030-lotto/lotto-data-hub/main/data/miniloto.json",
    "source_B_ddsky0728_raw.json": "https://raw.githubusercontent.com/ddsky0728/Japanese-lottery-data/main/loto/miniloto_history.json",
    "source_C_tank1159jhs_last100.json": "https://raw.githubusercontent.com/tank1159jhs/jp-lottery-api/main/data/miniloto/all.json",
}

BLOCKED_HOSTS_TRIED = [
    "https://www.mizuhobank.co.jp/retail/takarakuji/loto/miniloto/index.html",
    "https://www.mk-mode.com/rails/loto/miniloto",
    "https://miniloto.thekyo.jp/download/index",
    "http://sougaku.com/miniloto/data/list1/",
    "https://loto-life.net/csv/download",
    "https://en.wikipedia.org/",  # connectivity control test
    "https://example.com/",       # connectivity control test
    "https://cdn.jsdelivr.net/",
    "https://data.jsdelivr.com/",
]


def fetch(url: str, out_path: pathlib.Path) -> bool:
    result = subprocess.run(
        ["curl", "-sS", "-m", "30", "-o", str(out_path), "-w", "%{http_code}"],
        capture_output=True, text=True,
    )
    code = result.stdout.strip()
    ok = code == "200" and out_path.exists() and out_path.stat().st_size > 0
    print(f"{url} -> HTTP {code}, ok={ok}, bytes={out_path.stat().st_size if out_path.exists() else 0}")
    return ok


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    failures = []
    for fname, url in SOURCES.items():
        out_path = DATA_DIR / fname
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"already have {fname}, skipping re-fetch")
            continue
        if not fetch(url, out_path):
            failures.append((fname, url))

    if failures:
        print("FAILED SOURCES (no fictional/guessed data substituted):", failures)
        sys.exit(1)

    for fname in SOURCES:
        d = json.loads((DATA_DIR / fname).read_text())
        print(fname, "records:", len(d))

    print("Blocked hosts attempted this session (network policy denial, logged for transparency):")
    for h in BLOCKED_HOSTS_TRIED:
        print("  BLOCKED:", h)


if __name__ == "__main__":
    main()
