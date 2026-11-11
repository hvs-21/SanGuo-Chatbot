# SanGuo Chatbot

基于《三国演义》的本地问答演示。支持人物和事件查询、带回目和行号的原文检索，以及 TiRGN 与 RE-GCN 保存的预测案例。随仓库提供 120 回底本和 v17 的 588 条可引用图谱事实。

## 本地运行

安装 Python 3.10 或更高版本。从仓库根目录运行以下命令。

```text
python -B -X utf8 SanGuo-Chatbot/web_app.py --open-browser
```

浏览器访问 [本机问答页面](http://127.0.0.1:8765/)。Windows 也可双击 `SanGuo-Chatbot/启动演示.cmd`。基础演示使用 Python 标准库。建议先试「关羽和孙权是什么关系？」或「第 36 回，徐庶推荐了谁？」。

```text
cd SanGuo-Chatbot
python -B -X utf8 -m unittest -q test_web_app test_fact_qa test_query_graph test_predictions
python -B -X utf8 acceptance_check.py --port 8765
```

## 目录

- [SanGuo-Chatbot](SanGuo-Chatbot/) 包含网页、问答服务、可选接口和验收脚本。
- [sanguo_data_handoff](sanguo_data_handoff/) 包含 v17 事实表、实体、关系和时间数据。
- [sanguo_experiments](sanguo_experiments/) 包含原文、回目索引、保存的预测和实验报告。
- [结题材料](结题材料/) 包含全文问答验收题和基线。

公开版附带离线预测结果。实时 RE-GCN 补全和重新训练需要另行准备上游模型、运行环境和权重。千问整理可通过服务端环境变量 `DASHSCOPE_API_KEY` 配置。详细选项见 [使用说明](SanGuo-Chatbot/README.md)。

小说底本来自 [Project Gutenberg 23950](https://www.gutenberg.org/ebooks/23950)，随附的原始文件保留底本出处和许可说明。

## 早期版本说明

A QA system for Romance of the Three Kingdoms based on temporal knowledge graphs, RE-GCN, TiRGN, and RAG.
