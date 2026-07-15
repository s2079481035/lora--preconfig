"""
Step 4: ConfigBLEU Evaluation Metric (CLI Entry)
=================================================
Usage:
    python scripts/04_configbleu.py --demo
    python scripts/04_configbleu.py --candidate "..." --reference "..."
    python scripts/04_configbleu.py --eval-file data/test_results.json
"""

import sys, json, argparse, logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.configbleu import (
    compute_all_metrics, compute_configbleu,
    compute_bleu, compute_rouge_l, compute_meteor,
    ConfigASTParser, CONFIG_KEYWORDS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


def demo():
    print("=" * 60)
    print("ConfigBLEU Demonstration (论文 Figure 5 ACL 案例)")
    print("=" * 60)
    reference = "access-list 100 permit tcp 10.0.0.0 0.0.0.255 any eq 80"
    candidate_good = reference
    candidate_bad = "access-list 100 permit tcp 10.0.0.0 0.0.0.255 eq 80 any"

    print(f"\n参考配置: {reference}")
    print(f"正确候选: {candidate_good}")
    print(f"顺序错误: {candidate_bad}")

    for name, cand in [("正确", candidate_good), ("顺序错误", candidate_bad)]:
        print(f"\n--- {name} ---")
        result = compute_all_metrics(cand, reference)
        print(f"  BLEU:         {result['bleu']:.4f}")
        print(f"  BLEU_weight:  {result['bleu_weight']:.4f}")
        print(f"  Match_syn:    {result['match_syn']:.4f}")
        print(f"  ConfigBLEU:   {result['config_bleu']:.4f}")
        print(f"  ROUGE-L:      {result['rouge_l']:.4f}")
        print(f"  METEOR:       {result['meteor']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="ConfigBLEU Evaluation")
    parser.add_argument("--candidate", type=str, help="Candidate config text")
    parser.add_argument("--reference", type=str, help="Reference config text")
    parser.add_argument("--eval-file", type=str, help="JSON file with candidate/reference pairs")
    parser.add_argument("--demo", action="store_true", help="Run demonstration")
    args = parser.parse_args()

    if args.demo:
        demo()
        return

    if args.candidate and args.reference:
        result = compute_all_metrics(args.candidate, args.reference)
        print("\nEvaluation Results:")
        for k, v in result.items():
            print(f"  {k}: {v}")
    elif args.eval_file:
        with open(args.eval_file, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        for item in eval_data:
            result = compute_all_metrics(item["candidate"], item["reference"])
            print(f"\nCandidate: {item['candidate'][:50]}...")
            print(f"  ConfigBLEU: {result['config_bleu']}, BLEU: {result['bleu']}")
    else:
        demo()


if __name__ == "__main__":
    main()
