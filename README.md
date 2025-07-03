# AgentLaboratory

## 项目简介

AgentLaboratory 是一个全栈科研助理平台，集成了 PDF 文献管理、语义检索、自动化科研流程（文献综述、实验设计、数据处理、实验执行、结果分析、论文生成/翻译）等功能。后端基于 Flask，前端使用 Jinja2 + 原生 JavaScript，底层依赖 OpenAI / HuggingFace 大语言模型，可在 GPU（CUDA 12）或 CPU 环境下运行。

### 主要特性

1. **多用户会话隔离**：每位访客均分配独立 UUID 与专属目录 `user_data/`。
2. **PDF 上传与全文抽取**：支持批量上传，利用 PyPDF2 自动保存文本至 SQLite。
3. **语义搜索**：Sentence-Transformer 向量化 + 余弦相似度，毫秒级检索。
4. **浏览与批注**：内嵌 PDF 阅读器，可直接在浏览器查看原文。
5. **一键科研任务**：调用 `ai_lab_repo.LaboratoryWorkflow`，自动完成文献综述 → 方案制定 → 实验执行 → 结果解读 → 报告撰写 全流程，并支持 SSE 实时推送进度。
6. **可暂停 / 恢复**：阶段状态快照保存至磁盘，支持长时间任务容错。
7. **YAML 配置模板**：`experiment_configs/` 提供可复用的实验参数文件。
8. **GPU 加速**：默认拉取 CUDA 12 版 PyTorch，如无 GPU 可手动改装 CPU 版。

## 目录结构

```text
├── app.py                     # Web 入口 + REST API
├── ai_lab_repo.py             # 科研核心流水线
├── agents.py / *solver*.py     # 智能体与子任务解决器
├── templates/                 # Jinja2 HTML 模板
├── static/                    # 前端静态资源
├── uploads/                   # 公共上传目录
├── user_data/                 # 按用户 UUID 隔离的私人数据
├── Database/papers.db         # SQLite 数据库
├── experiment_configs/        # 预置 YAML 实验模板
├── requirements.txt           # Pip 依赖
└── environment.yaml           # Conda 依赖
```

## 环境准备

### 先决条件

* Python ≥ 3.10（推荐 3.12，与 `environment.yaml` 保持一致）
* (可选) NVIDIA GPU + CUDA 12，用于推理加速
* 可用的大模型 API-KEY（`OPENAI_API_KEY`、`DEEPSEEK_API_KEY` 等）

### 使用 Conda

```bash
conda env create -f environment.yaml
conda activate AgentLaboratory
```

### 使用 pip

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

> ⚠️ 若无 GPU，请先删除 `requirements.txt` 末尾三行 `torch* --index-url …`，再安装 CPU 版 PyTorch：
> ```bash
> pip install torch torchvision torchaudio
> ```

## 启动

```bash
export OPENAI_API_KEY=sk-...        # Windows: set OPENAI_API_KEY=...
python app.py --port 5200 --yaml-location experiment_configs/CIFAR_CNN.yaml
```

然后在浏览器打开 `http://localhost:5200`。

| 参数          | 说明                 | 默认值 |
|---------------|----------------------|--------|
| `--port`      | Web 服务器端口       | 5200   |
| `--yaml-location` | 默认实验配置文件 | `experiment_configs/CIFAR_CNN.yaml` |

## API 速览

| Method | Path | 描述 |
|--------|------|------|
| GET  | `/api/search?query=<关键字>` | 语义搜索论文 |
| POST | `/api/start_research` | 启动科研任务 |
| GET  | `/api/research_status/<task_id>` | 查询任务状态 |
| GET  | `/api/research_result/<task_id>` | 下载结果 ZIP |
| POST | `/api/pause_research/<task_id>` | 暂停任务 |
| POST | `/api/resume_research/<task_id>` | 恢复任务 |

请求示例：

```bash
curl -X POST http://localhost:5200/api/start_research \
  -H 'Content-Type: application/json' \
  -d '{
        "topic": "使用 Transformer 进行蛋白质结构预测",
        "paper_word_count": 4000,
        "config_template": "CIFAR_CNN.yaml",
        "language": "Chinese"
      }'
```

## 开发指南

1. **数据库**：模型定义于 `app.py`，修改后删除 `Database/papers.db` 即会自动重建。
2. **前端**：模板位于 `templates/`，静态资源放在 `static/`。Ajax 调用统一使用 `/api/*` 路径。
3. **新增科研子任务**：
   1. 在 `*solver*.py` 中实现 Solver 类；
   2. 在 `ai_lab_repo.LaboratoryWorkflow.phases` 与 `phase_models` 中注册；
   3. 更新 `app.py/run_research_task` 的映射逻辑。
4. **YAML 模板**：参考 `experiment_configs/blank.yaml`，字段均有中文注释。
5. **代码规范**：
   ```bash
   black .
   flake8
   ```
6. **测试**：
   ```bash
   pytest
   ```

## 常见问题

1. **页面空白 / 404**  
   检查浏览器控制台资源加载路径，确认 `static/`、`templates/` 不缺失。
2. **`OPENAI_API_KEY not set`**  
   请导出环境变量或在启动脚本中手动赋值 `os.environ["OPENAI_API_KEY"]`。
3. **CUDA 相关报错**  
   使用 CPU 版 PyTorch 或正确安装匹配版本的 CUDA 驱动。

## 许可证

本项目基于 Apache-2.0 许可证，详见 `LICENSE`。 