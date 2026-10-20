# 上游来源

模型与下列原始代码来自 [AlexWortega/openjev](https://huggingface.co/AlexWortega/openjev)，固定版本为 `a20448012c213128955ca0c693e7c943865cab77`。上游模型卡标注许可为 MIT。原作者文件保持原样：

- `sglang_client.py` ← `code/sglang_client.py`
- `vendor/openjev/decisions_api.py` ← `code/serving/decisions_api.py`
- `vendor/openjev/image_inputs.py` ← `code/serving/image_inputs.py`
- `vendor/openjev/serve_sglang.sh` ← `code/serve_sglang.sh`
- `vendor/openjev/sglang_openjev/` ← `code/sglang_openjev/`

作者的 [SGLang 说明](https://huggingface.co/AlexWortega/openjev/blob/a20448012c213128955ca0c693e7c943865cab77/README.md) 和 [答案规则](https://huggingface.co/AlexWortega/openjev/blob/a20448012c213128955ca0c693e7c943865cab77/code/serving/decisions_api.py) 是本仓库调用流程的依据。

图片配置来自 [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)，固定版本 `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`，由下载脚本获取，不提交模型或配置副本。

本仓库自己的部署管理、严格请求检查、多图示例和图片加速逻辑是服务适配代码，不应归为原作者 SGLang 实现。权重未修改；自定义服务会缩图，图片第一步改用相同参数的等价线性计算，同组图片特征在请求内复用，并使用可用的快速计算组件。因此不保证两套服务概率逐位相同。
