"""
Step 5: Model Fine-Tuning (LoRA + Instruction Tuning)
======================================================
对应论文 Section III-D: Fine-Tuning
- 基座模型: Qwen2.5-Coder-Instruct 1.5B
- 微调技术: LoRA (低秩适配)
- 训练方式: Instruction Tuning

使用方式:
    python scripts/05_finetune.py --data data/processed/train_data.json
    python scripts/05_finetune.py --data data/processed/train_data.json --eval-only
"""

import os
import json
import logging
import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def format_instruction(sample: Dict) -> Dict:
    """
    将数据格式化为 Instruction Tuning 格式
    论文: 使用 <指令, 输入, 输出> 三元组

    对应论文 Figure 1 的四种任务格式:
    - 配置生成: NL -> Config
    - 配置分析: Config -> NL
    - 配置翻译: Cisco Config -> Juniper Config
    - 配置补全: Partial Config -> Complete Config
    """
    task = sample.get("task", "config_generation")

    task_instructions = {
        "config_generation": "Generate the network configuration for the following requirement:",
        "config_analysis": "Analyze the following network configuration and describe its functionality in natural language:",
        "config_translation_c2j": "Translate the following Cisco IOS configuration to Juniper Junos format:",
        "config_translation_j2c": "Translate the following Juniper Junos configuration to Cisco IOS format:",
        "config_completion": "Complete the following incomplete network configuration with the missing lines:",
    }

    instruction = task_instructions.get(task, "Process the following network configuration:")

    # Build prompt in ChatML format for Qwen2.5
    messages = [
        {"role": "system", "content": "You are a network configuration expert. Generate accurate and syntactically correct network configurations."},
        {"role": "user", "content": f"{instruction}\n\n{sample.get('input', '')}"},
        {"role": "assistant", "content": sample.get("output", "")},
    ]

    return {"messages": messages}


def create_sample_training_data() -> List[Dict]:
    """Create sample training data for demonstration."""
    samples = [
        # Config Generation: NL -> Cisco
        {
            "task": "config_generation",
            "input": "Configure BGP on a Cisco router with AS 65000, establish a peering session with neighbor 192.168.1.1 in AS 64512, and advertise the network 10.0.0.0/24.",
            "output": """router bgp 65000
 neighbor 192.168.1.1 remote-as 64512
 address-family ipv4 unicast
  network 10.0.0.0 mask 255.255.255.0
  neighbor 192.168.1.1 activate
 exit-address-family""",
        },
        # Config Analysis: Cisco -> NL
        {
            "task": "config_analysis",
            "input": """router ospf 1
 network 10.0.0.0 0.0.0.255 area 0
 network 192.168.1.0 0.0.0.255 area 0
 default-information originate always""",
            "output": "This configuration enables OSPF routing process 1 on the router. It advertises two networks: 10.0.0.0/24 and 192.168.1.0/24, both in OSPF area 0 (backbone area). The 'default-information originate always' command causes the router to always advertise a default route (0.0.0.0/0) into the OSPF domain, even if the default route does not exist in the routing table.",
        },
        # Config Translation: Cisco -> Juniper
        {
            "task": "config_translation_c2j",
            "input": """interface GigabitEthernet0/0
 ip address 10.0.0.1 255.255.255.0
 no shutdown!
ip route 0.0.0.0 0.0.0.0 10.0.0.254""",
            "output": """interfaces {
    ge-0/0/0 {
        unit 0 {
            family inet {
                address 10.0.0.1/24;
            }
        }
    }
}
routing-options {
    static {
        route 0.0.0.0/0 {
            next-hop 10.0.0.254;
        }
    }
}""",
        },
        # Config Translation: Juniper -> Cisco
        {
            "task": "config_translation_j2c",
            "input": """protocols {
    ospf {
        area 0 {
            interface ge-0/0/0 {
                metric 10;
            }
            interface ge-0/0/1 {
                metric 20;
            }
        }
    }
}""",
            "output": """interface GigabitEthernet0/0
 ip ospf cost 10
!
interface GigabitEthernet0/1
 ip ospf cost 20
!
router ospf 1
 network 0.0.0.0 255.255.255.255 area 0""",
        },
        # Config Completion
        {
            "task": "config_completion",
            "input": """router bgp 65000
 neighbor 192.168.1.1 remote-as 64512
 address-family ipv4 unicast""",
            "output": """router bgp 65000
 neighbor 192.168.1.1 remote-as 64512
 address-family ipv4 unicast
  network 10.0.0.0 mask 255.255.255.0
  neighbor 192.168.1.1 activate
 exit-address-family""",
        },
        # More generation examples
        {
            "task": "config_generation",
            "input": "Set up OSPF on a Cisco router, enable it on interfaces connected to 10.0.0.0/24 and 172.16.0.0/16, both in area 0.",
            "output": """interface GigabitEthernet0/0
 ip ospf 1 area 0
!
interface GigabitEthernet0/1
 ip ospf 1 area 0
!
router ospf 1
 network 10.0.0.0 0.0.0.255 area 0
 network 172.16.0.0 0.0.255.255 area 0""",
        },
        {
            "task": "config_generation",
            "input": "Create an access list on Cisco that permits HTTP traffic from 10.0.0.0/24 to any destination and denies all other traffic.",
            "output": """ip access-list extended PERMIT_HTTP
 permit tcp 10.0.0.0 0.0.0.255 any eq 80
 deny ip any any""",
        },
        {
            "task": "config_generation",
            "input": "Configure a static route on Juniper to reach 172.16.0.0/16 via next-hop 10.0.0.254.",
            "output": """routing-options {
    static {
        route 172.16.0.0/16 {
            next-hop 10.0.0.254;
        }
    }
}""",
        },
        # More translation examples
        {
            "task": "config_translation_c2j",
            "input": """router bgp 65000
 bgp router-id 1.1.1.1
 neighbor 10.0.0.2 remote-as 65000
 address-family ipv4 unicast
  neighbor 10.0.0.2 next-hop-self""",
            "output": """routing-options {
    autonomous-system 65000;
}
protocols {
    bgp {
        group internal {
            type internal;
            local-address 1.1.1.1;
            neighbor 10.0.0.2;
        }
    }
}""",
        },
        {
            "task": "config_analysis",
            "input": """access-list 100 permit tcp 10.0.0.0 0.0.0.255 any eq 80
access-list 100 permit tcp 10.0.0.0 0.0.0.255 any eq 443
access-list 100 deny ip any any
interface GigabitEthernet0/0
 ip access-group 100 in""",
            "output": "This configuration implements an ingress access control list on interface GigabitEthernet0/0. ACL 100 permits HTTP (port 80) and HTTPS (port 443) traffic originating from the 10.0.0.0/24 subnet to any destination. All other IP traffic from this subnet is explicitly denied. This effectively restricts hosts on the 10.0.0.0/24 network to only web browsing traffic when entering through this interface.",
        },
    ]

    return samples


def prepare_training_data(data_path: Optional[str] = None, test_ratio: float = 0.0, test_output: Optional[str] = None):
    """Prepare training data in instruction format, optionally splitting test set."""
    if data_path and Path(data_path).exists():
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    else:
        logger.info("Using sample training data for demonstration")
        raw_data = create_sample_training_data()

    if test_ratio > 0:
        random.seed(42)
        indices = list(range(len(raw_data)))
        random.shuffle(indices)
        split_idx = int(len(raw_data) * (1 - test_ratio))
        train_raw = [raw_data[i] for i in indices[:split_idx]]
        test_raw = [raw_data[i] for i in indices[split_idx:]]
        logger.info(f"Split: {len(train_raw)} train, {len(test_raw)} test")

        if test_output:
            Path(test_output).parent.mkdir(parents=True, exist_ok=True)
            with open(test_output, "w", encoding="utf-8") as f:
                json.dump(test_raw, f, ensure_ascii=False, indent=2)
            logger.info(f"Test set saved to {test_output}")

        raw_data = train_raw

    formatted = [format_instruction(sample) for sample in raw_data]
    logger.info(f"Prepared {len(formatted)} training samples")
    return formatted


def run_finetuning(
    train_data: List[Dict],
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    output_dir: str = None,
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 5e-5,
    lora_r: int = 16,
):
    """
    论文 Section III-D: Fine-Tuning
    使用 LoRA + Instruction Tuning 微调 Qwen2.5-Coder
    """
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        from peft import LoraConfig, get_peft_model
        try:
            from trl import SFTTrainer, SFTConfig as TrainingArguments
        except ImportError:
            from transformers import TrainingArguments
            from trl import SFTTrainer
        from datasets import Dataset
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.info("Please install: pip install torch transformers peft trl datasets")
        logger.info("Saving formatted data for later training...")
        save_formatted_data(train_data, output_dir)
        return

    if output_dir is None:
        output_dir = str(MODEL_DIR / "preconfig-finetuned")

    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    if device == "cpu":
        logger.warning("No GPU detected. Training will be very slow.")

    # Load tokenizer
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    logger.info(f"Loading model: {model_name}")
    model_kwargs = {"trust_remote_code": True, "torch_dtype": torch.bfloat16, "device_map": "auto"}

    if device == "cuda" and getattr(torch.cuda.get_device_properties(0), 'total_memory', 0) < 8e9:
        logger.info("GPU memory < 8GB, using 4-bit quantization")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # 仅在启用量化时才调用 prepare_model_for_kbit_training
    if "quantization_config" in model_kwargs:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)

    # LoRA Configuration (论文 Section III-D.2)
    lora_config = LoraConfig(
        r=lora_r,                          # Low-rank dimension
        lora_alpha=lora_r * 2,             # Scaling factor
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"LoRA: {trainable_params:,} trainable / {total_params:,} total ({100*trainable_params/total_params:.2f}%)")

    # Prepare dataset
    dataset = Dataset.from_list(train_data)

    # Training arguments
    _ta_kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",
    )
    if hasattr(TrainingArguments, 'max_seq_length'):
        _ta_kwargs['max_seq_length'] = 2048
    training_args = TrainingArguments(**_ta_kwargs)

    # Trainer (兼容不同 trl 版本)
    trainer_kwargs = dict(model=model, args=training_args, train_dataset=dataset)
    try:
        trainer = SFTTrainer(**trainer_kwargs, tokenizer=tokenizer)
    except TypeError:
        trainer = SFTTrainer(**trainer_kwargs, processing_class=tokenizer)

    # Train
    logger.info("Starting fine-tuning...")
    trainer.train()

    # Save
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")


def save_formatted_data(train_data: List[Dict], output_dir: str = None):
    """Save formatted training data for later use."""
    if output_dir is None:
        output_dir = str(MODEL_DIR / "formatted_data")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    output_path = Path(output_dir) / "train_data.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Formatted training data saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="PreConfig Fine-Tuning")
    parser.add_argument("--data", type=str, default=None, help="Path to training data JSON")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--save-only", action="store_true", help="Only save formatted data, don't train")
    parser.add_argument("--test-ratio", type=float, default=0.0, help="Ratio of data to hold out as test set (default 0.0)")
    parser.add_argument("--test-output", type=str, default=None, help="Path to save test set JSON")
    args = parser.parse_args()

    # Auto-detect data if not specified
    data_path = args.data
    if data_path is None:
        candidates = [
            str(PROJECT_ROOT / "data" / "processed" / "train_data.json"),
            str(PROJECT_ROOT / "data" / "processed" / "forum_configs_extracted.json"),
        ]
        for c in candidates:
            if Path(c).exists():
                data_path = c
                logger.info(f"Auto-detected training data: {data_path}")
                break

    test_output = args.test_output
    if test_output is None and args.test_ratio > 0:
        test_output = str(PROJECT_ROOT / "data" / "processed" / "test_data.json")

    train_data = prepare_training_data(data_path, test_ratio=args.test_ratio, test_output=test_output)
    task_counts = {}
    for t in train_data:
        sample = t.get("messages", [{}])[-1] if "messages" in t else {}
        task = "unknown"
        for m in t.get("messages", []):
            if m.get("role") == "user":
                content = m["content"]
                if "Generate" in content: task = "generation"
                elif "Analyze" in content: task = "analysis"
                elif "Translate" in content: task = "translation"
                elif "Complete" in content: task = "completion"
        task_counts[task] = task_counts.get(task, 0) + 1
    logger.info(f"Training data composition: {task_counts}")

    if args.save_only:
        save_formatted_data(train_data, args.output)
        return

    # Run fine-tuning
    run_finetuning(
        train_data=train_data,
        model_name=args.model,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lora_r=args.lora_r,
    )


if __name__ == "__main__":
    main()
