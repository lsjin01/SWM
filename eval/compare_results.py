#!/usr/bin/env python3
"""
SWM vs WMPO vs SFT 결과 비교 테이블 출력.

사용법:
  python eval/compare_results.py --task square

또는 여러 JSON 파일 직접 지정:
  python eval/compare_results.py \\
    --files sft.json wmpo_p128.json wmpo_p1280.json swm.json
"""

import json
import argparse
from pathlib import Path


KNOWN_BASELINES = {
    "SFT":          0.24,
    "WMPO P_128":   0.22,
    "WMPO P_1280":  0.42,
    "Ours (30%)":   0.30,   # 기존 Option A 최고 결과
}


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def print_table(rows: list):
    """rows: list of (name, sr, n_success, n_episodes)"""
    print()
    print(f"{'Model':<25} {'SR':>8} {'Success':>10} {'Episodes':>10}")
    print("─" * 58)
    for name, sr, n_suc, n_ep in rows:
        bar = "█" * int(sr * 20) + "░" * (20 - int(sr * 20))
        print(f"{name:<25} {sr:>7.1%} {n_suc:>6}/{n_ep:<5} {bar}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",  default="square")
    parser.add_argument("--files", nargs="*", default=[],
                        help="JSON result files to compare")
    parser.add_argument("--eval_dir", default="eval_results",
                        help="디렉토리에서 JSON 자동 탐색")
    args = parser.parse_args()

    rows = []

    # Known baselines
    for name, sr in KNOWN_BASELINES.items():
        rows.append((name, sr, int(sr * 50), 50))

    # JSON 파일에서 로드
    files = args.files
    if not files:
        eval_dir = Path(args.eval_dir)
        if eval_dir.exists():
            files = sorted(eval_dir.glob(f"*{args.task}*.json"))

    for f in files:
        try:
            d = load_json(str(f))
            if d.get("task", "") == args.task or args.task in str(f):
                name     = Path(f).stem.replace("_", " ").title()
                sr       = d.get("success_rate", 0)
                n_suc    = d.get("n_success", int(sr * d.get("n_episodes", 50)))
                n_ep     = d.get("n_episodes", 50)
                ckpt     = d.get("stage3_ckpt", d.get("ckpt_path", ""))
                label    = f"{name} ({Path(ckpt).parent.name if ckpt else ''})"
                rows.append((label, sr, n_suc, n_ep))
        except Exception as e:
            print(f"  [warn] {f}: {e}")

    # 정렬 (SR 내림차순)
    rows.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'='*58}")
    print(f"  Task: {args.task}  —  Success Rate Comparison")
    print(f"{'='*58}")
    print_table(rows)


if __name__ == "__main__":
    main()
