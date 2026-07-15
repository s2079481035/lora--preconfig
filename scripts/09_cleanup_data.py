"""
Step 9: Cleanup Augmented Data
================================
Strips natural language contamination from LLM-augmented config data.

Two bugs fixed:
  1. Prompt allowed NL wrapping → model added "Note:...", "This config..."
  2. _llm_generate fallback split on ':' → captured NL lines instead of config

Usage:
    python scripts/09_cleanup_data.py
    python scripts/09_cleanup_data.py --input data/augmented/augmented_raw.json --output data/augmented/augmented_raw_clean.json
"""

import os, re, json, logging, argparse, random, shutil, datetime
from pathlib import Path
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent

# Cisco/Junper config line patterns
CONFIG_LINE_PATTERNS = re.compile(
    r'^(router\s|interface\s|access-list\s|line\s|snmp-server\s|'
    r'ip\s|arp\s|hostname\s|enable\s|username\s|banner\s|'
    r'transport\s|logging\s|ntp\s|clock\s|service\s|'
    r'set\s|show\s|run\s|edit\s|top\s|exit\s|commit\s|'
    r'protocols\s|routing-options\s|policy-options\s|'
    r'security\s|interfaces\s|system\s|'
    r'\}|\{|\s+\})',
    re.IGNORECASE
)

NL_MARKERS = re.compile(
    r'^(Note:|This\s+(variant|configuration|setup|version)|'
    r'Here\s+(is|are|we)|Below\s+is|The\s+(above|following)|'
    r'In\s+this|As\s+(an|a)|For\s+this|'
    r'Variant\s+\d+|'
    r'说明|以上|如下|示例|配置|注意|注：|这里|这个)',
    re.IGNORECASE
)


def is_config_line(line: str) -> bool:
    return bool(CONFIG_LINE_PATTERNS.match(line.strip()))


def is_nl_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if bool(NL_MARKERS.match(stripped)):
        return True
    # Lines that are pure prose (no config syntax)
    if (stripped[0].isupper() and stripped.endswith(('.', ':', '!', '?')) and
        not is_config_line(stripped)):
        return True
    return False


def strip_nl_wrapping(text: str) -> str:
    lines = text.split('\n')
    # Find first config line
    start = 0
    for i, line in enumerate(lines):
        if is_config_line(line):
            start = i
            break

    # Find last config line (scan from end)
    end = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        if is_config_line(lines[i]) or (lines[i].strip() and lines[i].strip() in ('}', ')', '{')):
            end = i + 1
            break

    cleaned = '\n'.join(lines[start:end])
    return cleaned.strip()


def fix_parsing_fragment(text: str) -> str:
    """Reconstruct config from fragments if it was split by buggy : fallback."""
    lines = text.split('\n')
    # If most lines are config lines, it's probably fine
    config_lines = [l for l in lines if is_config_line(l)]
    if len(config_lines) >= len(lines) * 0.5:
        return text

    # Try to find any config fragments and reconstruct
    # This is best-effort
    fragments = []
    for line in lines:
        stripped = line.strip()
        if is_config_line(stripped) or stripped in ('}', ')'):
            fragments.append(stripped)

    if fragments:
        return '\n'.join(fragments)

    return text


def is_valid_config(text: str) -> bool:
    """Check if text looks like a valid network config."""
    lines = text.strip().split('\n')
    if len(lines) < 2:
        return False
    config_count = sum(1 for l in lines if is_config_line(l.strip()))
    return config_count >= 1


def clean_augmented_data(data: List[Dict]) -> tuple:
    """Clean augmented data, return (cleaned, stats)."""
    cleaned = []
    stats = {
        'total': len(data),
        'llm_augmented': 0,
        'nl_removed': 0,
        'fragments_fixed': 0,
        'skipped_too_short': 0,
    }

    for item in data:
        doc_type = item.get('doc_type', '')
        cfg = item.get('config_text', '')
        nl = item.get('nl_text', '')

        # Only clean LLM-augmented items
        if 'augmented_deepseek' in doc_type or 'augmented_openai' in doc_type:
            stats['llm_augmented'] += 1

            # Strip NL wrapping
            cleaned_cfg = strip_nl_wrapping(cfg)
            if cleaned_cfg != cfg:
                stats['nl_removed'] += 1
                cfg = cleaned_cfg

            # Fix parsing fragments
            if not is_valid_config(cfg):
                cfg = fix_parsing_fragment(cfg)
                stats['fragments_fixed'] += 1

        if len(cfg) < 5 or len(nl) < 5:
            stats['skipped_too_short'] += 1
            continue

        item['config_text'] = cfg
        cleaned.append(item)

    stats['final'] = len(cleaned)
    return cleaned, stats


def recreate_instruction_data(augmented_raw: List[Dict], output_path: Path):
    """Recreate instruction-format data from cleaned augmented raw data."""
    samples = []
    for item in augmented_raw:
        nl = item.get("nl_text", "")
        cfg = item.get("config_text", "")
        if len(nl) < 5 or len(cfg) < 5:
            continue
        samples.append({
            "instruction": "Generate the network configuration for the following requirement:",
            "input": nl,
            "output": cfg,
            "task": "config_generation",
        })
        samples.append({
            "instruction": "Analyze the following network configuration and describe its functionality in natural language:",
            "input": cfg,
            "output": nl,
            "task": "config_analysis",
        })

    random.shuffle(samples)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    logger.info(f"Created {len(samples)} instruction samples → {output_path}")
    return samples


def _backup(path: Path, backup_dir: Path):
    """Create a timestamped backup of the file before overwriting."""
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{path.stem}_backup_{timestamp}{path.suffix}"
    shutil.copy2(path, backup_path)
    logger.info(f"Backup saved → {backup_path}")
    return backup_path


def main():
    parser = argparse.ArgumentParser(description="Clean NL contamination from augmented config data")
    parser.add_argument("--input", type=str,
                        default=str(PROJECT_ROOT / "data" / "augmented" / "augmented_raw.json"))
    parser.add_argument("--output", type=str,
                        default=str(PROJECT_ROOT / "data" / "augmented" / "augmented_raw.json"))
    parser.add_argument("--instruction-output", type=str,
                        default=str(PROJECT_ROOT / "data" / "processed" / "train_data_augmented.json"))
    parser.add_argument("--backup-dir", type=str,
                        default=str(PROJECT_ROOT / "data" / "backups"))
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        logger.error(f"Input not found: {in_path}")
        return

    with open(in_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    logger.info(f"Loaded {len(raw_data)} items from {in_path}")

    cleaned, stats = clean_augmented_data(raw_data)
    logger.info(f"Cleanup stats: {json.dumps(stats, indent=2)}")

    out_path = Path(args.output)
    backup_dir = Path(args.backup_dir)

    # Backup original before overwriting
    if out_path.exists():
        _backup(out_path, backup_dir)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved cleaned data → {out_path}")

    recreate_instruction_data(cleaned, Path(args.instruction_output))


if __name__ == "__main__":
    main()
