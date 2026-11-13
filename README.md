# AstrBot 智能分段插件

> v0.2.5 · 变更见 [CHANGELOG](CHANGELOG.md)

把主 LLM 回复拆成几条自然消息:先发首段,再按延迟异步补发剩余段。分段由 LLM 决定,超时或输出不可用时退回本地标点规则。

## 安装

放入 `data/plugins/` 后启用。运行时无第三方 Python 依赖。

## 配置

| 项 | 语义 |
| --- | --- |
| `enabled` | 总开关 |
| `provider_id` | 分段模型;留空用当前会话的聊天模型 |
| `style` | `natural` / `conservative` / `active` |
| `min_length` | 整条回复短于该长度时不拆 |
| `max_segments` | 单次最多条数 |
| `timeout_seconds` | 等模型超过该时间就改用本地规则 |
| `delay_base` / `delay_per_char` / `delay_max` | 补发间隔:基础延迟 + 字数 × 按字符延迟 + 固定 0~0.15 秒抖动,上限 `delay_max` |
| `streaming_compat_enabled` | 流式兼容,默认关 |
| `streaming_min_chars` / `streaming_max_chars` | 流式拆段的长度闸门 |

宿主内置分段回复(`platform_settings.segmented_reply.enable`)也开着时,插件让位给宿主并记一次 warning:宿主的切分在装饰钩子之后,会把本轮回复再切一遍,条数翻倍。

## 适用范围

- 处理 LLM / Agent 的纯文本结果,链形状须为「若干 At 前缀 + 末尾一条或多条 Plain」。带图片、引用或 At 在中间的链整条跳过,组件保持原位。
- 结果已由 `send_message_to_user` 发往当前会话时跳过分段:宿主靠它抑制重复回复,换链后宿主只比对首段。
- 补发走主动 `send_message`,不经回复管线的 after-hook(防递归)。依赖该 hook 的统计插件看不到补发段。
- 补发中途发送异常时,剩余段合并成一条发出;同一会话的多轮补发按序串行。
- 模型超时、解析失败或无可用模型时,用本地标点与括号规则兜底。本地规则只按标点与换行切:代码块与嵌套列表的行首缩进会被压平,ASCII 括号可能被当作动作括号拆开;流式兼容路径保留行首缩进。

## 流式兼容

默认关闭。开启后按标点与长度实时拆段,配置改动 30 秒内生效,无需重启。

- 平台按自身能力分级:WebChat 逐条发送,按插件边界生效;aiocqhttp、Line、Misskey、Slack、Mattermost 声明不支持流式消息,宿主会把每条链重新拼接再切分;飞书、Telegram、QQ 官方机器人、企业微信、Discord、微信公众号、Satori 会把各段累积成一条后发出。
- 补丁只挂在 `ResultDecorateStage`,不改各平台自己的流式发送。

## 模块结构

```
main.py         # Star 接线
settings.py     # 配置解析(每次 hook 重读)
bounds.py       # 配置上下限
segmenter.py    # 文本 → 分段(LLM + 本地兜底)
segmentation.py # 纯分段领域:文本规则 / LLM 契约 / 延迟
follow_up.py    # 补发调度与簿记
chain_utils.py  # MessageChain 工具
errors.py       # 异常摘要
streaming/      # 流式兼容(懒加载)
  patches.py    #   补丁装卸
  wrapper.py    #   实时断句
```

`main` 只做装配;`segmentation` 与 `settings` 只依赖 `bounds`,可以脱离 astrbot 单独测试。

## 开发

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

`requirements-dev.txt` 含 `pytest`、`pytest-asyncio`,以及 `metadata.yaml` 同版本的 `astrbot` 约束。

## 许可证

GPL-3.0-or-later
