# PreConfig 论文复现项目

## 项目结构

```
preconfig-reproduce/
├── configs/
│   └── config.yaml              # 全局配置（模型、训练参数、数据源）
├── data/
│   ├── raw/                     # 爬取的原始数据
│   ├── processed/               # 清洗后的任务数据
│   └── augmented/               # LLM 增强后的数据
├── scripts/
│   ├── 01_data_crawler.py       # 数据爬取（Cisco/Juniper 官网+论坛）
│   ├── 02_data_mining.py        # 数据挖掘（BoW+KNN / LLM翻译生成）
│   ├── 03_data_augmentation.py  # 数据增强（Prompt Engineering）
│   ├── 04_configbleu.py         # ConfigBLEU 评估指标实现
│   ├── 05_finetune.py           # 模型微调（LoRA + Instruction Tuning）
│   └── 06_reproduce_figures.py  # 复现论文所有图表
├── models/                      # 微调后的模型
├── figures/                     # 生成的图表
├── prompts/                     # Prompt 模板
└── requirements.txt             # Python 依赖
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 测试 ConfigBLEU 指标
python scripts/04_configbleu.py --demo

# 3. 生成论文所有图表（无需 GPU）
python scripts/06_reproduce_figures.py

# 4. 数据爬取（需要网络）
python scripts/01_data_crawler.py --source cisco --type docs

# 5. 数据挖掘
python scripts/02_data_mining.py --method bow_knn

# 6. 数据增强（需要 OpenAI API Key）
export OPENAI_API_KEY="your-key-here"
python scripts/03_data_augmentation.py --data data/processed/forum_configs_extracted.json

# 7. 模型微调（需要 GPU）
python scripts/05_finetune.py --save-only  # 先保存格式化数据
python scripts/05_finetune.py --data data/processed/train_data.json  # 开始训练
```

## 复现的论文图表

| 图表 | 内容 | 脚本 |
|:---|:---|:---|
| Table II | 训练数据集统计 | `06_reproduce_figures.py` |
| Table IV | 四任务评估结果 | `06_reproduce_figures.py` |
| Table V | PreConfig vs ChatGPT vs Gemini | `06_reproduce_figures.py` |
| Table VI | ConfigBLEU vs BLEU 对比 | `06_reproduce_figures.py` |
| Figure 2 | 框架 Pipeline 流程图 | `06_reproduce_figures.py` |
| Figure 5 | ACL ConfigBLEU 案例 | `06_reproduce_figures.py` |

## 需要掌握的关键知识

### 计算机网络
- OSPF 协议原理与配置语法
- BGP 协议原理与配置语法
- ACL（访问控制列表）配置
- Route-map（路由策略）配置
- Cisco IOS vs Juniper Junos 语法差异

### 机器学习/NLP
- Transformer 架构基础
- LoRA (Low-Rank Adaptation) 原理
- Instruction Tuning 训练范式
- BLEU / ROUGE / METEOR 评估指标
- TF-IDF + KNN 文本分类

### 工具链
- PyTorch + HuggingFace Transformers
- PEFT (Parameter-Efficient Fine-Tuning)
- BeautifulSoup (HTML 解析)
- matplotlib (数据可视化)
