# SanGuo-Chatbot
A QA system for Romance of the Three Kingdoms based on temporal knowledge graphs, RE-GCN, TiRGN, and RAG.

## 成品功能与当前进度

| 功能 | 成品如何使用 | 当前状态 |
| --- | --- | --- |
| 事实问答 | 查人物关系、事件和时间，回答附事实编号、文本片段和章节；找不到依据时明确提示 | 网页支持别名、事件方向及两跳关联路径 |
| 原文检索 | 按关键词查证据片段，或按回检索 120 回全文 | 本地网页可用，全文结果有底本、回数和行号 |
| 大模型整理 | 在有证据时可选用千问组织回答，仍显示原始依据，调用失败时回退 | 接口与回退通过模拟验证；真实 API 尚未验证 |
| 时态预测 | 选择人物、关系和时间案例，比较 TiRGN / RE-GCN 前五名候选，明确标记为模型预测 | 默认展示 v2 两份 seed 42 验证权重的离线案例 |
| 历史补全 | 给定回数、人物和动作，用 RE-GCN 实时排出候选，逐条核对原文 | 启用本地模型后，事实问答也会对适用的无结果查询做补全 |
| 实验与结题 | 保留可复跑配置、权重、日志、评估指标，以及系统测试、运行和答辩材料 | v2 三种子和频率基线的最终测试已保存，其他交付项见验收清单 |

事实问答与预测分开：事实来自有出处的语义主表；时态预测使用按时间切分的模型数据，不能把原文检索或测试集当作预测输入。

## 本地网页演示

Windows 安装 Python 3.10 或更高版本后，双击 `启动演示.cmd`。它会选一个空闲端口并打开浏览器。演示期间保留启动窗口，关闭服务后可双击重新启动。

在 `SanGuo-Chatbot` 目录启动服务：

```powershell
python .\web_app.py
```

浏览器打开 `http://127.0.0.1:8765/`，可用示例按钮试跑事实问答、证据片段和全文检索。多个关键词用空格分隔。全文支持加回数，例如 `第98回 王双 魏延`。Windows 使用简繁归一化匹配，引用保留底本字形。其他系统可按原文字形检索。端口被占用时可用 `--port 8766`。终端按 `Ctrl+C` 停止服务。

全文采用 [Project Gutenberg 23950 号中文本](https://www.gutenberg.org/ebooks/23950)。本地正文与章节索引通过文件摘要核对。索引来自 `sanguo_experiments/data_revisions/v2_fulltext_review/chapter_index.json`。Windows 简繁匹配使用系统的 [LCMapStringEx](https://learn.microsoft.com/zh-cn/windows/win32/api/winnls/nf-winnls-lcmapstringex)。

千问整理是可选功能。服务端从 `DASHSCOPE_API_KEY` 读取密钥，可以用 `SANGUO_QWEN_ENDPOINT` 和 `SANGUO_QWEN_MODEL` 覆盖接口地址与模型。按阿里云百炼的说明配置对应地域的接口，将密钥保留在服务端环境变量中。勾选后，千问可以整理图谱证据，或从本地检索到的原文段落归纳答案。原文回答会核对引文是否出自检索结果，并附出处供核对。证据不足、未开启或接口失败时，网页保留本地回答或线索。[阿里云百炼接口说明](https://www.alibabacloud.com/help/en/model-studio/qwen-api-via-openai-chat-completions)

独立验证网页接口：`python .\test_web_app.py`。测试中的千问响应由测试构造，不发送真实 API 请求。

### 网页验收题单

服务启动后，在同目录另开 PowerShell 执行 `python -B -X utf8 acceptance_check.py`。检查使用 [26 道验收题](acceptance_cases.json)，覆盖关系、事件、时间、别名、两跳路径、反向事件、资料不足、原文与社区检索，以及 v2 预测命中和未命中。若启动时改了端口，检查命令也加同样的 `--port`。典型例子是“云长何时斩杀华雄？”和“张飞和华雄有什么间接关联？”。

事实问答查不到时会尝试全文。以“曹操杀了刘备吗”为例，系统引用第 85 回的染病与驾崩段落；像“周瑜卧病时的体温是多少？”则展示相关原文线索，并说明结论无法确认。无密钥模式下运行 `python -B -X utf8 verify_fulltext_qa.py --port 8765` 可复查 30 道封存题并保存逐题结果。[全文问答验收说明](../结题材料/全文问答验收说明.md)

### 本地 BERT 语义解析

本地 BERT 补充了关系、两跳路径、人物事迹、事件时间及事件角色的部分改写问法。规则先处理已有问法。其余问题先通过别名表和结构约束确定人物、事件及方向，再将遮蔽人名后的问句与意图示例编码。只有最高的适用分数达到阈值且与其他候选拉开差距时才查询证据。分数只是余弦相似度。限制与歧义检查仍适用。

首次准备需联网下载约 96 MB 模型文件。权重、分词器及配置校验 SHA256 后落盘，后续在 CPU 离线加载。使用项目已有的 WSL Ubuntu、Python 3.9.25 与 PyTorch 2.1.0 时，可从 `SanGuo-Chatbot` 目录运行以下命令。安装目录与训练环境分离。

```powershell
python -B -X utf8 .\prepare_bert_model.py
wsl -d Ubuntu --cd '/path/to/checkout/SanGuo-Chatbot' -- /path/to/model-env/bin/python -m pip install --only-binary=:all: --target ../.runtime/bert/python -r bert_requirements.txt
$env:SANGUO_BERT_WSL_DISTRO = 'Ubuntu'
$env:SANGUO_BERT_PYTHON = '/path/to/model-env/bin/python'
python -B -X utf8 .\web_app.py --bert
```

迁移时将路径替换为实际 WSL 工作目录和带有 PyTorch 的解释器。Linux 本机可用 `SANGUO_BERT_PYTHON` 指定解释器。浏览器显示“本地 BERT 已启用”后可试问“华雄被云长斩杀是在什么时间？”。命中的答复显示“本地 BERT”、第 5 回及证据 F00167。出现反向角色、未覆盖的时间限定或无关问题时，系统会检查适用性和证据。

用 `python -B -X utf8 acceptance_check.py --bert` 运行 [12 道语义验收题](bert_acceptance_cases.json)，可同时加 `--backend neo4j`。BERT 初始化失败时网页仍可用规则解析，语义验收会报告功能未启用。模型为 BAAI 的 bge-small-zh-v1.5，使用其官方发布权重的 SafeTensors 版本。[模型说明](https://huggingface.co/BAAI/bge-small-zh-v1.5) · [权重摘要](https://huggingface.co/BAAI/bge-small-zh-v1.5/blob/main/model.safetensors)

### RE-GCN 历史补全

实时补全需要补全实验目录中的冻结配置、上游模型和权重。公开版随附离线预测供基础演示。

在已有模型实验环境的 Windows 机器上，从 `SanGuo-Chatbot` 目录启动：

```powershell
$env:SANGUO_COMPLETION_WSL_DISTRO = 'Ubuntu'
$env:SANGUO_COMPLETION_PYTHON = '/path/to/model-env/bin/python'
python -B -X utf8 .\web_app.py --completion --open-browser
```

请按实际环境替换解释器路径。也可同时加 `--bert`。Linux 可直接用已有 PyTorch 和 DGL 的解释器启动或设置 `SANGUO_COMPLETION_PYTHON`。推理在本地 CPU 使用已锁定的 RE-GCN 种子 42 权重和 460 条观测事实，启动时校验权重、代码和数据摘要。无需 API 密钥。

点击“历史补全”，或在事实问答输入“第6回，徐荣追击了谁？”。适用的问法要指定能唯一映射到已观测时间片的回数、明确人物和一个动作。事实问答优先采用已有证据，资料不足时再尝试补全。模型关系可能合并若干细粒度关系，原文核验依然要求人物、方向、细粒度关系、回数和时间片一致。

该例中曹操排第 4，对应复核记录 v2-B005 和底本第 994 行。其他候选显示“待核验”。没有直接证据时，事实答案显示“资料不足”。如果开启千问整理，只提交核验通过的证据。这条案例用来演示系统链路，模型成绩另见留出实验。[历史补全实验报告](../sanguo_experiments/runs/completion_v2_3seeds_r2/历史事实补全结果.md)

服务启动后，在另一个终端运行 `python -B -X utf8 acceptance_check.py --completion`。如同时开启 BERT 和 Neo4j，可加 `--bert --backend neo4j`。检查覆盖原文支持、全部待核验、已知事实优先、超出观测范围和时间映射歧义。

### 社区检索

选择“社区检索”并输入“云长”可查询关羽所在的社区，查看核心实体和原文支持的关系。网页读取随软件交付的 `community_index.json`，并校验它与当前事实的摘要及证据编号。摘要用固定模板根据实际社区成员和关系统计生成。

重建时使用 Python 3.9+、`igraph==0.11.9` 和 `texttable==1.7.0` 运行 `python -B build_communities.py`。Leiden 使用无向加权图，边权来自去重后的事实数，随机种子为 42，优化模块度。网页运行只需已有索引。算法参数见 [igraph 文档](https://python.igraph.org/en/0.11.9/api/igraph.Graph.html)。

### 本机 Neo4j 图谱

图谱使用带完整出处的问答事实，按内容摘要区分不同版本。每条边保存事实编号、关系、时间阶段、原文和章节。网页把已解析的实体、方向、关系和阶段作为参数执行 Cypher，支持两跳查询。原文检索与模型预测仍各自使用对应的数据源。

准备已有 Java 25 的 WSL Ubuntu。在 PowerShell 运行一次安装脚本，再启动项目专用的数据库：

```powershell
wsl -d Ubuntu --cd '/path/to/checkout/SanGuo-Chatbot' -- bash neo4j_runtime/setup.sh
wsl -d Ubuntu --cd '/path/to/checkout/SanGuo-Chatbot' -- bash neo4j_runtime/start.sh
```

安装从 Maven Central 获取 Neo4j Community 2026.10.2 和其运行库，保存到项目的 `.runtime/neo4j`。后续启动使用本地文件。项目实例的 HTTP 和 Bolt 分别监听 WSL 回环端口 `17474` 和 `17687`，这是无认证的本地演示实例。启动终端保留运行。

数据库启动后，在另一个 PowerShell 终端执行：

```powershell
cd 'D:\path\to\checkout\SanGuo-Chatbot'
$env:SANGUO_NEO4J_URL = 'http://127.0.0.1:17474'
$env:SANGUO_NEO4J_WSL_DISTRO = 'Ubuntu'
python -B -X utf8 .\neo4j_repository.py import
python -B -X utf8 .\web_app.py
```

导入会逐条核对数据库中的事实与原始证据。v15 的 Neo4j 验收覆盖 402 个节点、548 条事实。当前本地默认使用 v17 的 588 条事实。已有的 Neo4j 实例可重新运行导入命令，之后用 `python -B -X utf8 acceptance_check.py --backend neo4j` 核验。匹配的回答显示 Neo4j 标识。未启用数据库时，默认本地模式仍支持这些问法。

Windows 通过指定的 WSL 发行版查询回环地址，适用于本机端口没有自动转发的环境。接入 Windows 自己的 Neo4j 时，清除 `SANGUO_NEO4J_WSL_DISTRO`，使用 `SANGUO_NEO4J_USER`、`SANGUO_NEO4J_PASSWORD` 和可选的 `SANGUO_NEO4J_DATABASE` 环境变量。项目实例关闭了匿名使用报告、Fleet Manager 和实例发现。协议见 [Neo4j Query API](https://neo4j.com/docs/query-api/current/query/)。

页面下方“模型会怎样预测？”展示两种真实模型的验证集候选。选择一个案例即可比较前五名；页面不加载模型、不训练，也不调用千问。预测只使用该阶段之前的历史，按单步协议加入较早验证阶段的已发生事实。此处按原始模型得分排名，不用正确答案过滤候选，分数不表示正确概率。

可以把预测理解为填空题：遮住一条关系记录中的对象，让模型给候选排序。点击“揭晓数据集答案”，查看标注对象及其在各模型中的排名；切换题目会重新隐藏答案。相同题目的多个标注答案都会显示。答案在读取已保存预测后才附加，不参与打分或改动排名。标注仍需原文核对，没有命中标注不等于证明其他候选为假。

预测文件优先使用 `../sanguo_experiments/runs/formal_v2_3seeds_r1/validation_predictions.json`，尚未生成时可展示先导产物。可通过 `SANGUO_PREDICTIONS_PATH` 指定，v2 文件旁的 `prediction_data` 保存同版答案与映射。独立检查产物与异常处理：`python .\test_predictions.py`。复跑见 [实验结果与说明](../sanguo_experiments/runs/formal_v2_3seeds_r1/实验对照结果.md)。

## 本地证据问答

在 `SanGuo-Chatbot` 目录运行（仅用 Python 标准库，无需模型、API 或数据库）：

```powershell
python .\qa_pipeline.py '刘备和关羽在桃园结义阶段是什么关系？'
python .\qa_pipeline.py '曹操用什么手机？'
```

第一题应找到“结义兄弟”，同时显示事实编号、原文片段和《三国演义》章节。
第二题应提示资料不足，不返回无关的曹操资料。

也支持 `查询刘备`、`查询刘备在桃园结义阶段的事迹` 和 `刘备和关羽什么时候结义？`。
命令行按固定问法匹配人物、关系与时间，只使用主表中标为 `PASS` 且有完整来源的记录；
它不会对任意问法做推理，也不等同于全书检索。数据位置可通过 `--facts-file` 或 `SANGUO_FACTS_PATH` 指定。

运行独立测试：`python .\test_fact_qa.py`。

## 实体查询模块

查询模块会读取实体、关系、时间三张映射表，再依次扫描 `train.txt`、
`valid.txt`、`test.txt`，返回指定实体作为头实体或尾实体出现的全部事实。

直接查询：

```powershell
python query_graph.py 刘备
```

交互查询：

```powershell
python query_graph.py
```

运行独立测试：

```powershell
python test_query_graph.py
```

如数据不在默认位置，可以传入：

```powershell
python query_graph.py 刘备 --data-dir D:\path\to\SANGUO
```
