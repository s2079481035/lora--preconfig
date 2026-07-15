# 服务器部署指南

## 一、传输项目到服务器

### 方式一：scp 传输（推荐）

```bash
# 在本地 Windows 终端执行：
scp -r d:\study\RAG\network-config\preconfig-reproduce username@server_ip:~/preconfig-reproduce

# 或者先打包再传输（更快）
cd d:\study\RAG\network-config
tar -czf preconfig-reproduce.tar.gz preconfig-reproduce/
scp preconfig-reproduce.tar.gz username@server_ip:~/

# 在服务器上解压
ssh username@server_ip
tar -xzf preconfig-reproduce.tar.gz
```

### 方式二：git 传输

```bash
# 本地
cd d:\study\RAG\network-config\preconfig-reproduce
git init && git add -A && git commit -m "init"
git remote add origin username@server_ip:~/preconfig-reproduce.git

# 服务器
ssh username@server_ip
mkdir preconfig-reproduce.git && cd preconfig-reproduce.git
git init --bare

# 本地推送
git push -u origin main

# 服务器克隆
cd ~
git clone preconfig-reproduce.git
```

## 二、服务器环境配置

```bash
# SSH 登录服务器
ssh username@server_ip

# 进入项目目录
cd ~/preconfig-reproduce

# 运行部署脚本
bash deploy_server.sh
```

### 手动部署（如果脚本有问题）

```bash
# 1. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 2. 安装 PyTorch（根据 CUDA 版本选择）
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. 安装其他依赖
pip install transformers peft trl datasets accelerate
pip install beautifulsoup4 lxml scikit-learn
pip install matplotlib pandas numpy
pip install openai tqdm nltk rouge-score

# 4. 验证 GPU
python3 -c "import torch; print(torch.cuda.is_available())"
```

## 三、运行实验

### 后台运行（推荐）

```bash
# 使用 nohup 后台运行，断开 SSH 不会中断
nohup python3 scripts/01_data_crawler.py --all > logs/crawler.log 2>&1 &
echo $!  # 记录进程 ID

# 查看日志
tail -f logs/crawler.log

# 使用 screen/tmux（更推荐）
screen -S preconfig
# 或
tmux new -s preconfig

# 在 screen/tmux 中运行命令
python3 scripts/05_finetune.py

# 断开: Ctrl+A, D (screen) 或 Ctrl+B, D (tmux)
# 重新连接: screen -r preconfig 或 tmux attach -t preconfig
```

### GPU 训练命令

```bash
# 单 GPU 训练
python3 scripts/05_finetune.py \
    --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --epochs 3 \
    --batch-size 4 \
    --lr 5e-5

# 多 GPU 训练（如果有多张卡）
torchrun --nproc_per_node=2 scripts/05_finetune.py \
    --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --epochs 3 \
    --batch-size 8

# 显存不足时使用 4-bit 量化
python3 scripts/05_finetune.py \
    --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --epochs 3 \
    --batch-size 2
# 注意：代码会自动检测显存，< 8GB 时启用 4-bit 量化
```

## 四、下载模型（首次运行需要）

```bash
# 方式一：自动下载（需要能访问 HuggingFace）
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = 'Qwen/Qwen2.5-Coder-1.5B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
print('模型下载完成')
"

# 方式二：如果服务器无法访问 HuggingFace
# 在本地下载后上传
# 本地：
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-Coder-1.5B-Instruct --local-dir ./qwen2.5-coder-1.5b

# 上传到服务器：
scp -r ./qwen2.5-coder-1.5b username@server_ip:~/preconfig-reproduce/models/

# 修改 configs/config.yaml 中的 model.base_model 为本地路径
```

## 五、监控训练

```bash
# 查看 GPU 使用情况
nvidia-smi

# 实时监控
watch -n 1 nvidia-smi

# 查看训练日志
tail -f logs/training.log

# 查看进程
ps aux | grep python
```

## 六、下载结果到本地

```bash
# 从服务器下载微调后的模型
scp -r username@server_ip:~/preconfig-reproduce/models/preconfig-finetuned ./models/

# 下载图表
scp -r username@server_ip:~/preconfig-reproduce/figures ./figures/

# 下载日志
scp username@server_ip:~/preconfig-reproduce/logs/*.log ./logs/
```

## 七、常见问题

| 问题 | 解决方案 |
|:---|:---|
| CUDA out of memory | 减小 batch_size，或启用 4-bit 量化 |
| HuggingFace 下载慢 | 设置镜像: `export HF_ENDPOINT=https://hf-mirror.com` |
| pip 安装超时 | 使用国内镜像: `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple` |
| 权限不足 | `chmod +x deploy_server.sh` |
| Python 版本太低 | 使用 conda: `conda create -n preconfig python=3.10` |
