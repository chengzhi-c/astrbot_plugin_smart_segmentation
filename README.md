# AstrBot 智能分段插件

发送前把主 LLM 回复拆成更像真人聊天的多条消息：先发首段，再按延迟异步补发剩余段。

## 安装

放入 AstrBot 插件目录并启用。运行时**无第三方 Python 依赖**。

开发测试：

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

`requirements-dev.txt` 已包含 `pytest`、`pytest-asyncio` 与 `astrbot` 运行时版本约束（与 `metadata.yaml` 一致）。

## 配置要点

| 项 | 语义 |
| --- | --- |
| `enabled` | 总开关 |
| `provider_id` | 分段模型；空则用当前会话聊天模型 |
| `style` | `natural` / `conservative` / `active` |
| `min_length` | 短于该长度不分段 |
| `max_segments` | 单次最多条数 |
| `timeout_seconds` | 分段模型超时后改本地规则 |
| `delay_*` | 补发节奏 |
| `streaming_compat_enabled` | 流式兼容（默认关） |

建议关闭 AstrBot 内置 `platform_settings.segmented_reply.enable`，避免重复分段。

## 行为边界

- 只处理 LLM/Agent 的纯文本（及 At+Plain）结果。
- 模型超时、解析失败或无 provider 时走本地句读/括号规则兜底。
- 补发走主动 `send_message`，不经普通回复管线的 after-hook（防递归；依赖该 hook 的统计插件可能看不到补发段）。
- **流式兼容（C-light）**：开启后包装 AstrBot 流式结果并按标点/长度实时切段；**不再** patch 各平台 `send_streaming`。若某平台绕过 decorate 直发，流式分段不保证生效。
- 流式默认关闭时，不 import、不安装任何补丁（零成本）。

## 模块结构

```
main.py            # Star 接线（唯一入口）
settings.py        # 配置解析 + 缓存（bounds 对齐 schema）
bounds.py          # 配置上限单一来源
chain_utils.py     # MessageChain 工具
segmenter.py       # 文本 → 分段（LLM + 本地兜底）
follow_up.py       # 补发调度 + 事件簿记
segmentation.py    # 纯分段领域（文本规则 / LLM 契约 / 延迟）
errors.py          # 异常摘要
streaming/         # 可选流式兼容（懒加载；默认不装）
```

依赖方向单向：`main → segmenter → segmentation`，`follow_up → errors/segmentation`；`segmentation` 不依赖任何项目内模块。

## 许可证

GPL-3.0-or-later
