# Agent Laboratory: Using LLM Agents as Research Assistants


<p align="center">
  <img src="media/AgentLabLogo.png" alt="Demonstration of the flow of AgentClinic" style="width: 99%;">
</p>

<p align="center">
    【English | <a href="readme/README-chinese.md">中文</a> | <a href="readme/README-japanese.md">日本語</a> | <a href="readme/README-korean.md">한국어</a> | <a href="readme/README-filipino.md">Filipino</a> | <a href="readme/README-french.md">Français</a> | <a href="readme/README-slovak.md">Slovenčina</a> | <a href="readme/README-portugese.md">Português</a> | <a href="readme/README-spanish.md">Español</a> | <a href="readme/README-turkish.md">Türkçe</a> | <a href="readme/README-hindi.md">हिंदी</a> | <a href="readme/README-bengali.md">বাংলা</a> | <a href="readme/README-vietnamese.md">Tiếng Việt</a> | <a href="readme/README-russian.md">Русский</a> | <a href="readme/README-arabic.md">العربية</a> | <a href="readme/README-farsi.md">فارسی</a> | <a href="readme/README-italian.md">Italiano</a>】
</p>

<p align="center">
    【📝 <a href="https://arxiv.org/pdf/2501.04227">Paper</a> | 🌐 <a href="https://agentlaboratory.github.io/">Website</a> | 🌐 <a href="https://agentrxiv.github.io/">AgentRxiv Website</a> | 💻 <a href="https://github.com/SamuelSchmidgall/AgentLaboratory">Software</a> | 📰 <a href="https://agentlaboratory.github.io/#citation-ref">Citation</a>】
</p>

### News 
* [March/24/2025] 🎉 🎊 🎉 Now introducing **AgentRxiv**, a framework where autonomous research agents can upload, retrieve, and build on each other's research. This allows agents to make cumulative progress on their research.

## 📖 Overview

- **Agent Laboratory** is an end-to-end autonomous research workflow meant to assist **you** as the human researcher toward **implementing your research ideas**. Agent Laboratory consists of specialized agents driven by large language models to support you through the entire research workflow—from conducting literature reviews and formulating plans to executing experiments and writing comprehensive reports. 
- This system is not designed to replace your creativity but to complement it, enabling you to focus on ideation and critical thinking while automating repetitive and time-intensive tasks like coding and documentation. By accommodating varying levels of computational resources and human involvement, Agent Laboratory aims to accelerate scientific discovery and optimize your research productivity.
<p align="center">
  <img src="media/AgentLab.png" alt="Demonstration of the flow of AgentClinic" style="width: 99%;">
</p>

- Agent Laboratory also supports **AgentRxiv**, a framework where autonomous research agents can upload, retrieve, and build on each other's research. This allows agents to make cumulative progress on their research.

<p align="center">
  <img src="media/agentrxiv.png" alt="Demonstration of the flow of AgentClinic" style="width: 99%;">
</p>


### 🔬 How does Agent Laboratory work?

- Agent Laboratory consists of three primary phases that systematically guide the research process: (1) Literature Review, (2) Experimentation, and (3) Report Writing. During each phase, specialized agents driven by LLMs collaborate to accomplish distinct objectives, integrating external tools like arXiv, Hugging Face, Python, and LaTeX to optimize outcomes. This structured workflow begins with the independent collection and analysis of relevant research papers, progresses through collaborative planning and data preparation, and results in automated experimentation and comprehensive report generation. Details on specific agent roles and their contributions across these phases are discussed in the paper.

<p align="center">
  <img src="media/AgentLabWF.png" alt="Demonstration of the flow of AgentClinic" style="width: 99%;">
</p>


### 👾 Currently supported models

* **OpenAI**: o1, o1-preview, o1-mini, gpt-4o, o3-mini
* **DeepSeek**: deepseek-chat (deepseek-v3)

To select a specific llm set the flag `--llm-backend="llm_model"` for example `--llm-backend="gpt-4o"` or `--llm-backend="deepseek-chat"`. Please feel free to add a PR supporting new models according to your need!

## 🖥️ Installation

### Python venv option

* We recommend using python 3.12

1. **Clone the GitHub Repository**: Begin by cloning the repository using the command:
```bash
git clone git@github.com:SamuelSchmidgall/AgentLaboratory.git
```

2. **Set up and Activate Python Environment**
```bash
python -m venv venv_agent_lab
```
- Now activate this environment:
```bash
source venv_agent_lab/bin/activate
```

3. **Install required libraries**
```bash
pip install -r requirements.txt
```

4. **Install pdflatex [OPTIONAL]**
```bash
sudo apt install pdflatex
```
- This enables latex source to be compiled by the agents.
- **[IMPORTANT]** If this step cannot be run due to not having sudo access, pdf compiling can be turned off via running Agent Laboratory via setting the `--compile-latex` flag to false: `--compile-latex "false"`



5. **Now run Agent Laboratory!**

`python ai_lab_repo.py --yaml-location "experiment_configs/MATH_agentlab.yaml"`


### Co-Pilot mode

To run Agent Laboratory in copilot mode, simply set the copilot-mode flag in your yaml config to `"true"`

-----
## Tips for better research outcomes


#### [Tip #1] 📝 Make sure to write extensive notes! 📝

**Writing extensive notes is important** for helping your agent understand what you're looking to accomplish in your project, as well as any style preferences. Notes can include any experiments you want the agents to perform, providing API keys, certain plots or figures you want included, or anything you want the agent to know when performing research.

This is also your opportunity to let the agent know **what compute resources it has access to**, e.g. GPUs (how many, what type of GPU, how many GBs), CPUs (how many cores, what type of CPUs), storage limitations, and hardware specs.

In order to add notes, you must modify the task_notes_LLM structure inside of `ai_lab_repo.py`. Provided below is an example set of notes used for some of our experiments. 


```
task-notes:
  plan-formulation:
    - 'You should come up with a plan for only ONE experiment aimed at maximizing performance on the test set of MATH using prompting techniques.'
    - 'Please use gpt-4o-mini for your experiments'
    - 'You must evaluate on the entire 500 test questions of MATH'
  data-preparation:
    - 'Please use gpt-4o-mini for your experiments'
    - 'You must evaluate on the entire 500 test questions of MATH'
    - 'Here is a sample code you can use to load MATH\nfrom datasets import load_dataset\nMATH_test_set = load_dataset("HuggingFaceH4/MATH-500")["test"]'
...
```

--------

#### [Tip #2] 🚀 Using more powerful models generally leads to better research 🚀

When conducting research, **the choice of model can significantly impact the quality of results**. More powerful models tend to have higher accuracy, better reasoning capabilities, and better report generation. If computational resources allow, prioritize the use of advanced models such as o1-(mini/preview) or similar state-of-the-art large language models.

However, **it's important to balance performance and cost-effectiveness**. While powerful models may yield better results, they are often more expensive and time-consuming to run. Consider using them selectively—for instance, for key experiments or final analyses—while relying on smaller, more efficient models for iterative tasks or initial prototyping.

When resources are limited, **optimize by fine-tuning smaller models** on your specific dataset or combining pre-trained models with task-specific prompts to achieve the desired balance between performance and computational efficiency.

-----

#### [Tip #3] ✅ You can load previous saves from checkpoints ✅

**If you lose progress, internet connection, or if a subtask fails, you can always load from a previous state.** All of your progress is saved by default in the `state_saves` variable, which stores each individual checkpoint. 

-----


#### [Tip #4] 🈯 If you are running in a language other than English 🈲

If you are running Agent Laboratory in a language other than English, no problem, just make sure to provide a language flag to the agents to perform research in your preferred language. Note that we have not extensively studied running Agent Laboratory in other languages, so be sure to report any problems you encounter.

For example, if you are running in Chinese set the language in the yaml:

`language:  "中文"`

----


#### [Tip #5] 🌟 There is a lot of room for improvement 🌟

There is a lot of room to improve this codebase, so if you end up making changes and want to help the community, please feel free to share the changes you've made! We hope this tool helps you!


## 📜 License

Source Code Licensing: Our project's source code is licensed under the MIT License. This license permits the use, modification, and distribution of the code, subject to certain conditions outlined in the MIT License.

## 📬 Contact

If you would like to get in touch, feel free to reach out to [sschmi46@jhu.edu](mailto:sschmi46@jhu.edu)

## Reference / Bibtex


### Agent Laboratory
```bibtex
@misc{schmidgall2025agentlaboratoryusingllm,
      title={Agent Laboratory: Using LLM Agents as Research Assistants}, 
      author={Samuel Schmidgall and Yusheng Su and Ze Wang and Ximeng Sun and Jialian Wu and Xiaodong Yu and Jiang Liu and Zicheng Liu and Emad Barsoum},
      year={2025},
      eprint={2501.04227},
      archivePrefix={arXiv},
      primaryClass={cs.HC},
      url={https://arxiv.org/abs/2501.04227}, 
}
```

### AgentRxiv
```bibtex
@misc{schmidgall2025agentrxiv,
      title={AgentRxiv: Towards Collaborative Autonomous Research}, 
      author={Samuel Schmidgall and Michael Moor},
      year={2025},
      eprint={2503.18102},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2503.18102}, 
}
```

## 🔄 多用户支持与文件管理

Agent Laboratory现已支持多用户并行使用和隔离的文件管理系统，特别适合需要构建网站界面供多人使用的场景。

### 主要改进：

1. **用户会话管理**：
   - 为每个用户创建唯一的会话ID
   - 所有用户数据关联到其唯一标识符
   - 自动创建用户专属数据目录

2. **安全的文件隔离**：
   - 每个用户的数据、代码和生成的图片严格隔离
   - 防止用户间数据泄露或干扰
   - 自动权限检查确保用户只能访问自己的数据

3. **改进的文件管理**：
   - 所有生成的图像和文件保存在用户自己的目录中
   - 解决了之前图片直接保存在根目录的问题
   - 为matplotlib绘图添加自动重定向功能

4. **研究任务API**：
   - 新增API端点用于启动研究任务
   - 支持跟踪研究进度和获取结果
   - 异步执行长时间运行的实验

5. **定期清理机制**：
   - 自动清理过期的用户数据和任务
   - 优化存储空间使用
   - 可配置的数据保留策略

### 技术实现：

- 使用Flask会话管理用户状态
- 修改文件生成逻辑，包括matplotlib绘图函数
- 添加用户权限验证层
- 实现异步任务处理机制
- 建立数据库模型追踪用户和任务

### 使用方法：

用户现在可以通过网站界面提交研究请求，系统会自动为其创建独立的工作环境，并保存所有研究成果在用户专属目录中。用户可以随时查看研究进度、下载报告和图表，而无需担心与其他用户的数据混淆。

# AgentLaboratory Web服务器集成

本项目实现了AgentLaboratory研究自动化框架与Web服务器的集成，使用户可以通过Web界面轻松启动和管理自动化研究任务。

## 主要功能

- 通过Web界面管理研究任务
- 支持多种语言的研究报告生成
- 用户数据管理和会话控制
- 论文上传和搜索功能
- 研究结果可视化和共享
- 与AgentRxiv集成，支持自主研究代理之间的协作

## 技术改进

### 配置整合

我们对`app.py`进行了修改，整合了`ai_lab_repo.py`中的配置逻辑，主要包括：

1. 添加命令行参数支持，允许用户指定端口和配置文件：
   ```python
   parser = argparse.ArgumentParser(description="AgentLaboratory Web Server")
   parser.add_argument('--port', type=int, default=5000, help='Web服务器监听端口')
   parser.add_argument('--yaml-location', type=str, default="experiment_configs/MATH_agentlab.yaml", help='YAML配置文件路径，用于加载默认配置')
   ```

2. 从YAML配置文件加载默认设置，并存储在app.config中：
   ```python
   app.config['DEFAULT_LLM_BACKBONE'] = config.get('llm-backend', "o4-mini-yunwu")
   app.config['DEFAULT_LANGUAGE'] = config.get('language', '中文')
   app.config['DEFAULT_NUM_PAPERS_LIT_REVIEW'] = config.get('num-papers-lit-review', 5)
   # 更多配置...
   ```

3. 支持API密钥的环境变量设置：
   ```python
   api_key = config.get('api-key')
   if api_key and not os.environ.get('OPENAI_API_KEY'):
       os.environ['OPENAI_API_KEY'] = api_key
   ```

### AgentRxiv支持

为了支持AgentRxiv功能，我们进行了以下改进：

1. 修改了`AgentRxiv`类，使其能够接受端口参数：
   ```python
   def __init__(self, lab_index=0, port=None):
       self.port = port if port is not None else 5000 + self.lab_index
   ```

2. 在`run_app`函数中初始化全局AgentRxiv实例：
   ```python
   if AI_LAB_AVAILABLE:
       from ai_lab_repo import AgentRxiv
       import ai_lab_repo
       ai_lab_repo.GLOBAL_AGENTRXIV = AgentRxiv(lab_index=app.config.get('DEFAULT_LAB_INDEX', 0), port=port)
   ```

3. 在`run_research_task`函数中确保AgentRxiv正确初始化：
   ```python
   if agentRxiv and not ai_lab_repo.GLOBAL_AGENTRXIV:
       lab_index = workflow_params.get('lab_index', 0)
       ai_lab_repo.GLOBAL_AGENTRXIV = AgentRxiv(lab_index=lab_index)
   ```

### 研究任务处理改进

我们对`run_research_task`函数进行了全面改进：

1. 使用app.config中的默认值作为配置回退：
   ```python
   if 'num-papers-lit-review' in config:
       workflow_params['num_papers_lit_review'] = config['num-papers-lit-review']
   elif 'DEFAULT_NUM_PAPERS_LIT_REVIEW' in app.config:
       workflow_params['num_papers_lit_review'] = app.config['DEFAULT_NUM_PAPERS_LIT_REVIEW']
   ```

2. 改进了任务笔记的处理逻辑，支持多语言：
   ```python
   # 处理任务笔记，转换为LaboratoryWorkflow可接受的格式
   task_notes_LLM = []
   task_notes = config['task-notes']
   
   # 收集所有实际涉及的任务阶段
   phases_in_notes = set()
   
   for _task in task_notes:
       readable_phase = _task.replace("-", " ")
       phases_in_notes.add(readable_phase)
       for _note in task_notes[_task]:
           task_notes_LLM.append({"phases": [readable_phase], "note": _note})
   ```

3. 添加了agent模型配置：
   ```python
   llm_backend = config.get('llm-backend', app.config.get('DEFAULT_LLM_BACKBONE', 'o4-mini-yunwu'))
   agent_models = {
       "literature review": llm_backend,
       "plan formulation": llm_backend,
       # 更多阶段...
   }
   workflow_params['agent_model_backbone'] = agent_models
   ```

## 使用方法

启动服务器：
```bash
python app.py --port 5000 --yaml-location "experiment_configs/MATH_agentlab.yaml"
```

访问Web界面：
```
http://localhost:5000
```

## 注意事项

- 确保已安装所有必要的依赖项
- 在启用AgentRxiv功能时，确保服务器在正确的端口上运行
- 对于非英语研究，请在配置文件中设置适当的语言参数