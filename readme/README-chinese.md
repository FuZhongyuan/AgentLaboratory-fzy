# fzy Report Agent: 使用大型语言模型代理作为研究助理

<p align="center">
  <img src="../media/AgentLabLogo.png" alt="Demonstration of the flow of AgentClinic" style="width: 99%;">
</p>

<p align="center">
    【<a href="../README.md">English</a>  | 中文 | <a href="../readme/README-japanese.md">日本語</a> | <a href="../readme/README-korean.md">한국어</a> | <a href="../readme/README-filipino.md">Filipino</a> | <a href="../readme/README-french.md">Français</a> | <a href="../readme/README-slovak.md">Slovenčina</a> | <a href="../readme/README-portugese.md">Português</a> | <a href="../readme/README-spanish.md">Español</a> | <a href="../readme/README-turkish.md">Türkçe</a> | <a href="../readme/README-hindi.md">हिंदी</a> | <a href="../readme/README-bengali.md">বাংলা</a> | <a href="../readme/README-vietnamese.md">Tiếng Việt</a> | <a href="../readme/README-russian.md">Русский</a> | <a href="../readme/README-arabic.md">العربية</a> | <a href="../readme/README-farsi.md">فارسی</a> | <a href="../readme/README-italian.md">Italiano</a>】
</p>

<p align="center">
    【🌐 <a href="https://agentlaboratory.github.io/">网站</a> | 💻 <a href="https://github.com/SamuelSchmidgall/AgentLaboratory">软件</a> | 🎥 <a href="https://agentlaboratory.github.io/#youtube-video">视频</a> |  📚 <a href="https://agentlaboratory.github.io/#examples-goto">示例论文</a> | 📰 <a href="https://agentlaboratory.github.io/#citation-ref">引用</a>】
</p>

## 📖 概述

- **fzy Report Agent** 是一个端到端的自主研究工作流程，旨在协助**您**作为人类研究人员**实现您的研究想法**。fzy Report Agent 由由大型语言模型驱动的专业代理组成，支持您完成整个研究工作流程——从进行文献综述和制定计划，到执行实验和撰写综合报告。
- 该系统并非旨在取代您的创造力，而是为了补充它，使您能够专注于创意和批判性思维，同时自动化重复且耗时的任务，如编码和文档编写。通过适应不同水平的计算资源和人类参与，fzy Report Agent 旨在加速科学发现并优化您的研究生产力。

<p align="center">
  <img src="../media/AgentLab.png" alt="Demonstration of the flow of AgentClinic" style="width: 99%;">
</p>

### 🔬 fzy Report Agent 如何工作？

- fzy Report Agent 包含三个主要阶段，系统地引导研究过程：（1）文献综述，（2）实验，（3）报告撰写。在每个阶段，由大型语言模型驱动的专业代理协作完成不同的目标，整合了如 arXiv、Hugging Face、Python 和 LaTeX 等外部工具以优化结果。这一结构化的工作流程始于独立收集和分析相关研究论文，经过协作计划和数据准备，最终实现自动化实验和综合报告生成。论文中讨论了具体代理角色及其在这些阶段的贡献。

<p align="center">
  <img src="../media/AgentLabWF.png" alt="Demonstration of the flow of AgentClinic" style="width: 99%;">
</p>

## 🖥️ 安装


### Python 虚拟环境选项

1. **克隆 GitHub 仓库**：首先使用以下命令克隆仓库：
    ```bash
    git clone git@github.com:SamuelSchmidgall/AgentLaboratory.git
    ```

2. **设置并激活 Python 环境**
    ```bash
    python -m venv venv_agent_lab
    ```
    - 现在激活此环境：
    ```bash
    source venv_agent_lab/bin/activate
    ```

3. **安装所需库**
    ```bash
    pip install -r requirements.txt
    ```

4. **安装 pdflatex [可选]**
    ```bash
    sudo apt install pdflatex
    ```
    - 这使得代理能够编译 latex 源代码。
    - **[重要]** 如果由于没有 sudo 权限而无法运行此步骤，可以通过将 `--compile_latex` 标志设置为 false 来关闭 pdf 编译：`--compile_latex=False`

5. **现在运行 fzy Report Agent！**
    
    `python ai_lab_repo.py --api-key "API_KEY_HERE" --llm-backend "o1-mini" --research-topic "YOUR RESEARCH IDEA"`
    
    或者，如果您没有安装 pdflatex
    
    `python ai_lab_repo.py --api-key "API_KEY_HERE" --llm-backend "o1-mini" --research-topic "YOUR RESEARCH IDEA" --compile_latex=False`

-----

## 提高研究成果的技巧

#### [技巧 #1] 📝 确保写下详尽的笔记！ 📝

**写下详尽的笔记非常重要**，帮助您的代理理解您在项目中希望实现的目标，以及任何风格偏好。笔记可以包括您希望代理执行的任何实验、提供 API 密钥、希望包含的特定图表或图形，或任何您希望代理在进行研究时了解的内容。

这也是您让代理知道**它可以访问的计算资源**的机会，例如 GPU（数量、类型、内存大小）、CPU（核心数量、类型）、存储限制和硬件规格。

为了添加笔记，您必须修改 `ai_lab_repo.py` 中的 `task_notes_LLM` 结构。以下是我们的一些实验中使用的笔记示例。

```
task_notes_LLM = [
    {"phases": ["plan formulation"],
     "note": f"You should come up with a plan for TWO experiments."},

    {"phases": ["plan formulation", "data preparation",  "running experiments"],
     "note": "Please use gpt-4o-mini for your experiments."},

    {"phases": ["running experiments"],
     "note": f"Use the following code to inference gpt-4o-mini: \nfrom openai import OpenAI\nos.environ["OPENAI_API_KEY"] = "{api_key}"\nclient = OpenAI()\ncompletion = client.chat.completions.create(\nmodel="gpt-4o-mini-2024-07-18", messages=messages)\nanswer = completion.choices[0].message.content\n"},

    {"phases": ["running experiments"],
     "note": f"You have access to only gpt-4o-mini using the OpenAI API, please use the following key {api_key} but do not use too many inferences. Do not use openai.ChatCompletion.create or any openai==0.28 commands. Instead use the provided inference code."},

    {"phases": ["running experiments"],
     "note": "I would recommend using a small dataset (approximately only 100 data points) to run experiments in order to save time. Do not use much more than this unless you have to or are running the final tests."},

    {"phases": ["data preparation", "running experiments"],
     "note": "You are running on a MacBook laptop. You can use 'mps' with PyTorch"},

    {"phases": ["data preparation", "running experiments"],
     "note": "Generate figures with very colorful and artistic design."},
    ]
```

--------

#### [技巧 #2] 🚀 使用更强大的模型通常会带来更好的研究 🚀

在进行研究时，**模型的选择会显著影响结果的质量**。更强大的模型往往具有更高的准确性、更好的推理能力和更优秀的报告生成能力。如果计算资源允许，优先使用先进的模型，如 o1-(mini/preview) 或类似的最先进大型语言模型。

然而，**在性能和成本效益之间取得平衡也很重要**。虽然强大的模型可能会产生更好的结果，但它们通常更昂贵且运行时间更长。考虑选择性地使用它们，例如用于关键实验或最终分析，同时在迭代任务或初步原型设计中依赖较小、更高效的模型。

当资源有限时，**通过在您的特定数据集上微调较小的模型或将预训练模型与特定任务的提示相结合来优化，以实现性能与计算效率之间的理想平衡**。

-----

#### [技巧 #3] ✅ 您可以从检查点加载之前的保存 ✅

**如果您丢失了进度、互联网连接中断或子任务失败，您始终可以从先前的状态加载。** 您的所有进度默认保存在 `state_saves` 变量中，该变量存储每个单独的检查点。只需在运行 `ai_lab_repo.py` 时传递以下参数

`python ai_lab_repo.py --api-key "API_KEY_HERE" --research-topic "YOUR RESEARCH IDEA" --llm-backend "o1-mini" --load-existing True --load-existing-path "save_states/LOAD_PATH"`

-----

#### [技巧 #4] 🈯 如果您使用非英语语言运行 🈲

如果您使用非英语语言运行 fzy Report Agent，没问题，只需确保向代理提供一个语言标志，以便用您喜欢的语言进行研究。请注意，我们尚未广泛研究使用其他语言运行 fzy Report Agent，因此请务必报告您遇到的任何问题。

例如，如果您使用中文运行：

`python ai_lab_repo.py --api-key "API_KEY_HERE" --research-topic "YOUR RESEARCH IDEA (in your language)" --llm-backend "o1-mini" --language "中文"`

----

#### [技巧 #5] 🌟 还有很大的改进空间 🌟

这个代码库还有很大的改进空间，因此如果您进行了更改并希望帮助社区，请随时分享您所做的更改！我们希望这个工具对您有帮助！

## 参考文献 / Bibtex

```bibtex
@preprint{schmidgall2025AgentLaboratory,
  title={fzy Report Agent: Using LLM Agents as Research Assistants},
  author={Schmidgall, Samuel and Su, Yusheng and Wang, Ze and Sun, Ximeng and Wu, Jialian and Yu, Xiadong and Liu, Jiang, Liu, Zicheng and Barsoum, Emad},
  year={2025}
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

## 📚 论文版本自动切换功能

为了提高研究过程中论文获取的成功率，系统现在支持在论文查询失败时自动尝试获取同一论文的不同版本。

### 主要特性：

1. **自动版本切换**：
   - 当论文查询达到最大重试次数(5次)后仍然失败时，系统会自动尝试查询同一论文的其他版本
   - 支持在带版本号的论文ID(如1611.05431v3)和不带版本号的ID之间智能切换

2. **版本遍历**：
   - 对于带版本号的论文ID，系统会尝试v1到v5的所有版本
   - 对于不带版本号的论文ID，系统会尝试添加v1后缀进行查询

3. **错误处理优化**：
   - 针对常见的论文获取错误(如"object has no attribute 'updated_parsed'")提供了解决方案
   - 减少因特定版本不可用导致的研究中断

4. **用户友好提示**：
   - 当成功获取到替代版本时，系统会清晰标明原始版本和替代版本信息
   - 提供详细的错误日志，便于问题排查

这一功能显著提高了系统获取研究论文的稳定性，特别是在处理arXiv上的论文时，确保研究流程不会因单个论文版本的获取问题而中断。

## 🔧 修复状态保存序列化问题

本次修改解决了在保存研究状态时出现的 "Can't pickle local object 'state_callback_wrapper.<locals>.callback'" 错误。

### 问题描述

在多用户环境下，系统使用 pickle 模块来序列化和保存研究任务的状态，以便在需要时恢复。然而，当研究任务中包含局部定义的回调函数作为回调时，pickle 无法序列化这些函数，导致状态保存失败。

### 解决方案

1. **创建全局回调包装器**：
   - 添加了全局变量 `_TASK_ID_FOR_CALLBACK` 用于存储任务ID
   - 创建了模块级别的 `global_state_callback` 函数，可以被pickle序列化
   - 修改 `state_callback_wrapper` 函数，使其返回全局函数而不是局部函数

2. **回调机制重构**：
   - 将 `run_research_task` 和 `continue_research_task` 函数中的回调机制与全局回调系统集成
   - 保持参数传递的一致性，通过全局变量传递任务ID

3. **保持接口一致**：
   - 确保新的回调系统与原有系统接口完全一致
   - 对使用方无需任何调整，透明替换

### 效果

这些修改确保了：

1. 研究任务状态可以正确序列化和保存
2. 用户可以暂停和恢复研究任务而不会遇到序列化错误
3. 系统可靠性大幅提高，特别是在长时间运行的研究任务中

## 代码运行错误修复

针对代码执行过程中出现的 `Expected more than 1 value per channel when training, got input size torch.Size([1, 512, 1, 1])` 错误，我们实施了以下优化：

### 问题描述

深度学习实验中，当批处理大小（batch size）为1时，使用BatchNormalization层会导致错误，因为该层需要至少两个样本来计算统计数据。

### 解决方案

1. **批处理大小检查**：
   - 添加了自动检测批处理大小的机制
   - 当检测到包含BatchNorm并且批处理大小为1时，自动添加修复代码

2. **BatchNorm层修复**：
   - 添加 `ensure_batchnorm_works` 辅助函数，当批处理大小为1时自动将所有BatchNorm层设置为eval模式
   - 在模型加载后自动应用此修复

3. **执行超时优化**：
   - 将代码执行的默认超时时间从600秒增加到1200秒，以适应更复杂的实验

### 效果

1. **提高稳定性**：自动修复BatchNorm问题，避免实验因批处理大小错误而失败
2. **更好的兼容性**：确保生成的代码在各种模型架构和数据集大小下正常工作
3. **降低用户调试负担**：系统自动处理常见错误，使用户能够专注于研究内容
4. **支持更长时间的执行**：通过增加超时限制，支持更复杂的实验和数据处理

这些改进使Agent Laboratory在处理深度学习实验时更加稳健，特别是在处理小数据集或需要长时间计算的场景中。

## 文件生成路径优化

针对用户报告的"文件生成在根目录"问题，我们进行了以下优化：

### 问题描述

尽管系统有多用户支持和文件隔离机制，但生成的代码文件（如generated_code.py）和下载的数据集仍然出现在应用根目录，而不是用户特定目录中。

### 解决方案

1. **工作目录管理改进**：
   - 优化了工作目录的切换机制，确保所有文件操作都在用户目录中执行
   - 使用绝对路径保存生成的代码文件，避免路径混淆

2. **数据集下载路径重定向**：
   - 重定向了常见机器学习库的缓存目录到用户特定的`.cache`目录
   - 针对Hugging Face、PyTorch、TensorFlow等库设置了专用环境变量
   - 为每个用户创建独立的缓存结构，避免数据交叉污染

3. **路径转换增强**：
   - 改进了相对路径到绝对路径的转换逻辑
   - 确保生成的图表和其他输出文件保存在正确的用户目录中

4. **环境变量管理**：
   - 在代码执行前临时修改环境变量，重定向数据和模型存储路径
   - 执行完成后自动恢复原始环境变量，避免对系统产生持久影响

### 效果

1. **文件隔离**：所有生成的文件（代码、数据集、模型、图表）现在都保存在用户特定目录中
2. **更好的隐私保护**：每个用户的数据完全隔离，不会互相干扰
3. **清晰的目录结构**：用户可以在自己的目录中找到所有相关文件，提高可管理性
4. **根目录整洁**：应用根目录不再被临时文件污染，提高系统稳定性和安全性

这些改进解决了文件生成位置不正确的问题，特别是在多用户环境下，确保每个用户的研究成果都保存在正确的位置，并可以被适当地访问和管理。

## 优化TensorFlow和机器学习库的日志输出

为了提高用户体验和减少不必要的日志输出，我们对系统中的TensorFlow和其他机器学习库的警告和消息进行了抑制处理。

### 问题描述

在执行机器学习代码时，TensorFlow和其他深度学习框架会产生大量的初始化消息、警告和提示，这些包括：
1. 设备信息 (CPU/GPU)
2. 优化提示
3. 库初始化消息
4. 已弃用功能警告
5. oneDNN加速提示

这些消息会在控制台中占用大量空间，干扰用户查看实际的执行结果和输出，尤其在多用户环境下运行多个任务时更为明显。

### 解决方案

1. **环境变量控制**：
   - 通过设置`TF_CPP_MIN_LOG_LEVEL=3`将TensorFlow日志级别提高到仅显示错误
   - 禁用oneDNN自定义操作提示通过`TF_ENABLE_ONEDNN_OPTS=0`
   - 配置其他库的环境变量以减少冗余输出

2. **Python警告抑制**：
   - 使用`warnings.filterwarnings('ignore')`全局禁用Python警告
   - 对特定库进行更精细的警告控制

3. **日志级别管理**：
   - 设置TensorFlow和Keras的日志级别为ERROR
   - 配置相关库的日志处理器以过滤低级别消息

4. **输出捕获与过滤**：
   - 使用上下文管理器临时捕获并过滤库的初始化输出
   - 确保只有用户关心的输出被显示

5. **代码生成优化**：
   - 在生成的代码中自动添加警告抑制逻辑
   - 确保用户执行代码时也不会看到大量重复的警告

### 效果

1. **输出整洁性**：终端输出现在更加简洁明了，只显示与用户任务直接相关的信息
2. **更好的用户体验**：减少了用户在大量警告消息中寻找重要输出的麻烦
3. **日志质量提升**：通过只保留重要的警告和错误，提高了日志的信息密度
4. **减轻服务器负担**：在多用户环境下，减少了不必要的控制台输出处理开销

这些优化使Agent Laboratory在处理TensorFlow和其他机器学习库时更加友好，特别适合在教育环境或多用户研究平台中使用。

## 修复代码执行输出目录为None的问题

我们解决了一个重要的问题，即在某些情况下代码执行时输出目录被设置为`None`，导致文件被保存在根目录而不是用户特定目录中。

### 问题描述

在执行实验代码时，系统日志会显示"执行代码的目标输出目录: None"的消息，这表明`execute_code`函数没有接收到正确的目录参数。这导致生成的代码文件（如`generated_code.py`）和数据集下载被保存在应用程序的根目录中，而不是用户特定的目录中。

主要原因是在以下几个地方调用`execute_code`函数时没有传递必要的`user_id`或`lab_dir`参数：

1. `mlesolver.py`中的`Replace`类和`Edit`类
2. `MLESolver`类的`run_code`方法
3. 创建`MLESolver`实例时未传递`lab_dir`参数

### 解决方案

1. **参数传递完善**：
   - 修改了所有调用`execute_code`的地方，确保传递`lab_dir`参数
   - 在`MLESolver`类中添加了`lab_dir`属性，并在创建实例时传递该参数

2. **智能目录检测**：
   - 当`lab_dir`参数未提供时，添加了智能检测逻辑
   - 首先检查当前目录是否为用户目录
   - 然后尝试从环境变量获取输出目录
   - 最后使用当前工作目录作为后备选项

3. **环境变量管理**：
   - 确保环境变量`CURRENT_OUTPUT_DIR`被正确设置和使用
   - 在代码执行完成后保存输出目录信息，以便后续操作使用

### 效果

1. **正确的文件保存位置**：所有生成的代码文件和下载的数据集现在都保存在用户特定的目录中
2. **一致的工作目录**：确保代码执行时使用一致的工作目录，避免路径混淆
3. **更好的多用户支持**：增强了多用户环境下的文件隔离
4. **清晰的日志信息**：添加了更详细的日志，显示文件保存位置，便于调试和跟踪

这些修改确保了系统在执行代码时能够正确处理文件路径，无论是在单用户环境还是多用户环境中，都能保持文件的正确组织和隔离。

## 图像生成与保存优化

为了提高研究结果的可视化效果和持久性，我们对代码生成指令进行了优化，确保在实验过程中生成的图像能够被正确保存和展示。

### 问题描述

在之前的版本中，模型生成的代码可能没有明确的指导来保存生成的图像，导致：
1. 可视化结果在代码执行完成后丢失
2. 无法在报告中引用和展示实验生成的图表
3. 缺少对图像命名和组织的统一标准

### 解决方案

1. **可视化指导增强**：
   - 在MLEngineerAgent、SWEngineerAgent和PhDStudentAgent的命令描述中添加了专门的可视化指导
   - 提供了详细的图像生成和保存最佳实践

2. **图像保存标准化**：
   - 要求使用plt.savefig()保存所有生成的图像，并使用描述性文件名
   - 建议在保存后使用plt.show()确保图像在执行过程中也能被显示
   - 鼓励添加适当的标题、标签和图例，提高图像的信息价值

3. **多样化可视化建议**：
   - 鼓励创建多个可视化图表，展示数据或结果的不同方面
   - 对于交互式可视化，确保同时保存为静态图像，保证兼容性

### 效果

1. **研究结果持久化**：实验生成的所有图像现在都会被保存到用户目录中，便于后续查看和分析
2. **报告质量提升**：保存的图像可以直接被引用到研究报告中，提高报告的可视化质量
3. **可视化标准化**：统一的图像命名和格式化标准，使研究成果更加专业和一致
4. **用户体验改进**：研究人员可以更方便地查看、比较和分享实验结果

这些改进确保了在实验过程中生成的所有可视化内容都能被正确保存，并且可以在研究报告中被有效利用，大大提高了研究成果的质量和可重复性。