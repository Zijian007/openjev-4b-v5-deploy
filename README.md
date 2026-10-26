# OpenJev 4B v5 部署

在 Linux 上部署 [AlexWortega/openjev 的 4B v5 模型](https://huggingface.co/AlexWortega/openjev/tree/main/qwen3.5-4b-nli-v5)，支持文字、图片和多个问题。仓库仅包含代码、小型示例图片和依赖清单，**不包含模型、运行环境或个人服务器信息**。

| 服务 | 默认端口 | 调用方式 | 图片处理 |
| --- | --- | --- | --- |
| 自定义服务 | 31000 | `requests` 直接提交 `state`、`images`、`questions` | 最长边限制为 1280；输入上限 4096 |
| 作者 SGLang 服务 | 30000 | 作者的 `OpenJevSGLang`，通过 `/classify` 调用 | 使用 SGLang/Qwen 原有图片处理；输入上限 8192 |

两套服务共用下载的模型权重，各自使用独立环境。自定义服务保留作者的问题模板和答案规则，增加多图格式检查、任务取消与恢复，以及图片计算加速；SGLang 的模型支持代码与客户端保持作者原样。两边图片输入和计算方式不同，答案与概率可能不同。

## 1. 准备机器与克隆

服务端要求 **Linux x86_64、NVIDIA 显卡和可正常使用的 NVIDIA 驱动**。建议使用 24 GB 显存显卡，实际需求取决于图片和并发量。客户端可以是 Linux、Windows 或 macOS。

模型约 9.1 GB，两套环境分别约 7.6 GB、9.3 GB；安装还需要下载和缓存空间，建议至少预留 50 GB。自定义环境使用 CUDA 12.8 版 Torch；SGLang 环境使用 CUDA 13.0 版 Torch，因此运行 SGLang 需支持 CUDA 13.0 的驱动。先运行 `nvidia-smi` 检查显卡和驱动。

```bash
git clone https://github.com/yuechuan-ma/openjev-4b-v5-deploy.git
cd openjev-4b-v5-deploy
```

安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)（若已安装可跳过）：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

下面命令均在仓库根目录执行，不需要 Docker 或 sudo。环境脚本会使用或下载 Python 3.12，并安装清单中固定版本的依赖。

## 2. 配置环境与下载模型

按需要安装其中一套，或分别安装两套：

```bash
# 自定义服务环境：.venv
bash scripts/setup.sh custom

# SGLang 环境：.venv-sglang
bash scripts/setup.sh sglang
```

脚本最后会检查显卡基础计算。首次安装会下载较多文件；不要在服务运行时重新安装它的环境。

自定义服务可以额外安装快速卷积组件。需要系统已安装 **与 Torch CUDA 12.8 兼容的 CUDA Toolkit、nvcc 和 C++ 编译工具**；编译工具缺失时使用基础环境即可，但图片处理可能更慢：

```bash
bash scripts/setup.sh custom --accelerate
```

依赖清单位于 `requirements/custom.lock.txt`、`requirements/sglang.lock.txt`；可选组件为 `causal-conv1d==1.7.0`。不要把两套依赖装进同一环境。

**模型只需下载一次**，用任一已安装环境执行：

```bash
# 安装了自定义环境时
.venv/bin/python download_model.py

# 如果只安装 SGLang，改用：
# .venv-sglang/bin/python download_model.py
```

脚本下载固定版本 `a20448012c213128955ca0c693e7c943865cab77` 下的 `qwen3.5-4b-nli-v5`，并验证模型与图片配置的校验值。断点下载后可重复执行。无需手动修改模型配置。

- `upstream/qwen3.5-4b-nli-v5/`：原始模型。
- `model-overlay/`：两套服务读取的目录，链接原始权重并补充 Qwen 原版图片配置，不复制第二份权重。
- `MODEL_SOURCE.json`：下载版本记录。

这些目录和文件均已被 Git 忽略。

## 3. 启动、停止和查看服务

### 自定义服务：31000

```bash
.venv/bin/python service.py start
.venv/bin/python service.py status
.venv/bin/python service.py stop
.venv/bin/python service.py restart
.venv/bin/python service.py logs
```

### 作者 SGLang 服务：30000

```bash
.venv-sglang/bin/python scripts/sglang/service_sglang.py start
.venv-sglang/bin/python scripts/sglang/service_sglang.py status
.venv-sglang/bin/python scripts/sglang/service_sglang.py stop
.venv-sglang/bin/python scripts/sglang/service_sglang.py restart
.venv-sglang/bin/python scripts/sglang/service_sglang.py logs
```

以上是独立操作命令，按需执行。两套 `start` 都会等待模型就绪才返回；首次启动 SGLang 可能需要准备计算组件。关闭 SSH 窗口后服务继续运行，**没有设置开机自动启动**。停止时会关闭本服务的计算子进程；正常退出超时后会强制结束本服务。

两套服务默认都使用 **GPU 0**。单张显卡建议每次运行一套；两套同时运行需要分别留足显存。多显卡可在启动或重启时指定，例如：

```bash
OPENJEV_GPU=0 .venv/bin/python service.py start
SGLANG_GPU=1 .venv-sglang/bin/python scripts/sglang/service_sglang.py start
```

SGLang 默认显存预算为整张显卡的 60%，可用 `SGLANG_MEM_FRACTION=0.60` 调整；启动等待时间默认 600 秒，可用 `SGLANG_START_TIMEOUT` 调整。变更显卡等设置后需重启相应服务。

```bash
# 健康检查
curl -f --max-time 5 http://127.0.0.1:31000/health
curl -f --max-time 5 http://127.0.0.1:30000/health

# 持续查看日志，Ctrl+C 只退出日志查看
tail -f logs/service.log
tail -f logs/sglang.log
```

## 4. 本机与其他机器调用

调用代码在服务器上运行时，地址用 `127.0.0.1`；在其他机器上运行时，改成**服务器可访问的 IP 或主机名**，并确保相应端口可达。服务监听 `0.0.0.0`，两套服务接口不通用。

调用端不需要显卡或模型。克隆仓库后，在调用端自己的 Python 环境安装：

```bash
python -m pip install requests numpy
```

### 31000：直接用 requests

```python
import requests

server_ip = "127.0.0.1"  # 跨机器时改成服务器 IP
body = {
    "state": "A man is playing a guitar.",
    "questions": {
        "music": {"type": "noul", "instructions": "Is someone making music?"}
    }
}
response = requests.post(
    f"http://{server_ip}:31000/v1/systemone", json=body, timeout=600
)
response.raise_for_status()
print(response.json())
```

完整的 body、三种问题、图片路径转换、共享多图与错误说明见 **[REQUEST_FORMAT.md](REQUEST_FORMAT.md)**。

```bash
# 本机使用已安装的环境运行示例
.venv/bin/python examples/example_text.py
.venv/bin/python examples/example_image.py
.venv/bin/python examples/example_multi_image.py

# 其他机器：将 SERVER_IP 换为服务器地址
python examples/example_multi_image.py --host SERVER_IP --port 31000
```

### 30000：作者 OpenJevSGLang

```python
from scripts.sglang.sglang_client import OpenJevSGLang

client = OpenJevSGLang("http://127.0.0.1:30000")  # 跨机器时改成服务器 IP
print(client.predict([
    ("A man is playing a guitar.", "Someone is making music.")
]))  # 顺序：contradiction、entailment、neutral
```

调用代码应在仓库根目录运行，以便导入作者客户端。两图例子使用作者模板和答案规则，每个问题共同参考两张图：

```bash
.venv-sglang/bin/python examples/example_multi_image_sglang.py
python examples/example_multi_image_sglang.py --host SERVER_IP --port 30000
```

两份多图例子默认都使用仓库内的红、蓝纯色图，问题完全相同。可用 `--images first.jpg second.jpg` 换成**运行调用代码的电脑上的路径**。上传的是图片内容，服务器不读取客户端路径。发送给 SGLang 的图片使用带格式前缀的 data URI，避免 JPG 数据被误当成文件路径。

## 检查与运维

31000 就绪后可运行 `.venv/bin/python scripts/check_service.py`，检查多图和请求格式。`scripts/check_recovery.py` 会中断并重启 31000，只在专门排查恢复问题时手动运行。安装任一环境后可运行无需模型的仓库检查，例如 `.venv/bin/python -m unittest discover -s tests`。

模型、模板和原作者代码来源及固定版本见 [THIRD_PARTY.md](THIRD_PARTY.md)。本仓库未重新训练或更换权重，图片处理与计算优化不保证与另一套服务的每个概率完全相同。
