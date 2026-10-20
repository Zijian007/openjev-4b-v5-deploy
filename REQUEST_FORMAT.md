# 请求格式

## 31000：POST /v1/systemone

使用 `requests.post(url, json=body, timeout=600)`，不要把 body 写进 `data=`。所有问题共享同一份 `state` 和全部图片。

```json
{
  "state": "Image 0: <<IMG[0]>>\nImage 1: <<IMG[1]>>",
  "images": [{"data": "图片0的base64内容"}, {"data": "图片1的base64内容"}],
  "questions": {
    "color": {
      "type": "choice",
      "instructions": "What are the dominant colors of Image 0 and Image 1?",
      "criteria": {"A": "Red and blue", "B": "Blue and red"}
    },
    "both_colors": {"type": "noul", "instructions": "Do these images include both red and blue?"},
    "similarity": {
      "type": "score",
      "instructions": "How similar are the images?",
      "criteria": ["Different", "Somewhat similar", "Very similar"]
    }
  }
}
```

上面图片内容是占位说明，实际发送必须换成有效图片数据。

### 共享字段

| 字段 | 格式与规则 |
| --- | --- |
| `state` | 必填字符串，最长 110000 字符。有图片时可为空；无图片时不能为空白 |
| `images` | 可省略或为 `[]`。最多 4 项，每项只能包含 `data`，没有 `id` |
| `images[i].data` | 图片文件内容的 base64 字符串，或 `data:image/jpeg;base64,...` 等 data URI；不能是路径或网址 |
| `questions` | 必填非空字典，最多 64 个问题；键为非空问题名称，不能重复 |

只接受这三个顶层字段，不填写 `model`、`image_data` 或 `id`。问题只接受 `type`、`instructions`、`criteria`；不接受未知字段或重复 JSON 字段。

### 三种问题

每个问题的 `instructions` 必须是非空字符串。

| type | criteria | 返回 |
| --- | --- | --- |
| `choice` | 非空字典，键为选项名，值为非空描述；最多 64 项 | `choice` 为选项名，另有 `probabilities` 和 `confidence` |
| `noul` | 可省略，默认 no/yes；自定义时只能是 `{"false": "否的描述", "true": "是的描述"}` | `noul` 为“是”的概率，范围 0～1 |
| `score` | 从低到高排列的非空描述列表，最多 64 项 | 第一个等级为 0，依次递增；`score` 为概率加权分数，另有 `probabilities`、`legend` 和 `confidence` |

一次请求总计最多 128 个候选答案，一个 `noul` 计两个。模型逐个判断候选答案，再按作者 `normalized_entailment_v1` 规则整理；`confidence` 是根据选项分布计算的参考值，不保证答案正确。

### 图片下标与容量

- 下标从 0 开始，`<<IMG[0]>>` 对应 `images[0]`。
- 标记只能出现在 `state`；问题通过“Image 0”等文字指代图片。
- 可调换引用顺序，也可重复引用；不存在的下标报错。
- 未引用的图片自动附在 state 后，注明下标；有图片时允许 state 为空。
- 不接受旧标记 `<<IMG>>` 或用户写入的模型控制标记。
- 图片支持单帧 JPG、PNG、WebP；每张原始文件最多 4 MiB、1200 万像素；整个请求最多 24 MiB。
- 31000 会先按比例缩小到最长边不超过 1280，不裁剪、不放大小图；这可能改变细节与判断。
- 文字和图片的完整输入上限为 4096。图片越大、重复引用越多，占用越多。超过限制明确报错，不自动截断；最多 4 张不代表任意 4 张都能放下。

### 从路径构造图片数据

路径属于调用端电脑，上传文件内容。下面同时适用于本机与跨机器调用：

```python
import base64
import mimetypes
from pathlib import Path
import requests

server_ip = "127.0.0.1"  # 跨机器时替换为服务器 IP
images = []
for path in [Path("examples/red.png"), Path("examples/blue.png")]:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    images.append({"data": f"data:{mime};base64,{encoded}"})

body = {
    "state": "Image 0: <<IMG[0]>>\nImage 1: <<IMG[1]>>",
    "images": images,
    "questions": {
        "order": {
            "type": "choice",
            "instructions": "What are the colors of Image 0 and Image 1, in that order?",
            "criteria": {"A": "Red and blue", "B": "Blue and red"}
        }
    }
}
response = requests.post(
    f"http://{server_ip}:31000/v1/systemone", json=body, timeout=600
)
response.raise_for_status()
print(response.json()["answers"]["order"])
```

### 错误和等待

| HTTP 状态 | 含义 |
| --- | --- |
| 400 | 格式错误、图片损坏、下标不存在等；查看错误指出的位置 |
| 413 | 请求、图片或完整输入超限 |
| 503 | 繁忙、显存不足或正在恢复；减少输入或稍后重试 |
| 504 | 计算超过默认 180 秒，已取消并开始恢复 |

错误示例：

```json
{"error": {"code": 400, "message": "state: <<IMG[1]>> references an image that does not exist"}}
```

最多 4 个正在计算或等待的请求，排队最长默认 30 秒。调用断开后，未开始的请求取消；已开始的计算会结束相应计算进程并重载模型。恢复期间 `/health` 为 HTTP 503，恢复后为 HTTP 200。

## 30000：SGLang POST /classify

**SGLang 不接收上面的 body**。它接收已经按作者模板展开的文字和图片，返回三类原始分数。推荐使用原作者 `sglang_client.py`，或直接使用 requests：

```python
import requests

server_ip = "127.0.0.1"
body = {"text": ["Premise: A man is playing a guitar.\nHypothesis: Someone is making music."]}
response = requests.post(f"http://{server_ip}:30000/classify", json=body, timeout=600)
response.raise_for_status()
print(response.json())  # 各条结果的 embedding 是 contradiction、entailment、neutral 三个分数
```

包含图片时，`text` 中每张图的位置使用 `<|vision_start|><|image_pad|><|vision_end|>`。`image_data` 的外层列表对应各条文字，内层列表对应这条文字引用的全部图片。用 data URI 发送，避免 JPG 的 `/9j/` 开头被识别成文件路径。例如：

```python
body = {
    "text": [
        "Premise: Image 0: <|vision_start|><|image_pad|><|vision_end|>\n"
        "Image 1: <|vision_start|><|image_pad|><|vision_end|>\n"
        "Hypothesis: The images include both red and blue."
    ],
    "image_data": [[images[0]["data"], images[1]["data"]]]
}
```

上例的 `images` 来自前面的路径转换代码。多个候选判断需要多条 `text`，每条都携带它需要参考的图片。仓库 `example_multi_image_sglang.py` 已演示共享多图、多个问题，以及使用作者规则把原始分数整理成答案；31000 的字段检查和 4 张图片限制不应套用到原版 SGLang 接口。
