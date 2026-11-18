# AstrBot 智能分段插件

> v0.2.7 · 变更见 [CHANGELOG](CHANGELOG.md)

把主 LLM 回复拆成几条自然消息：先发首段，再按延迟异步补发剩余段。分段由 LLM 决定，超时或输出不可用时退回本地标点规则。

## 安装

需要 AstrBot `>= 4.16, < 5`。放入 `data/plugins/` 后启用，运行时无第三方 Python 依赖。

## 配置

| 项 | 语义 |
| --- | --- |
| `enabled` | 总开关 |
| `provider_id` | 分段模型；指定时整条回复会发送给该模型，留空用当前会话的聊天模型 |
| `style` | `natural` / `conservative` / `active` |
| `min_length` | 整条回复短于该长度时不拆 |
| `max_segments` | 目标分段数量；内容装不下时可能超过该值 |
| `timeout_seconds` | 等模型超过该时间就改用本地规则 |
| `delay_base` / `delay_per_char` / `delay_max` | 补发间隔：基础延迟 + 字数 × 按字符延迟 + 0~0.15 秒随机抖动，上限 `delay_max` |
| `streaming_compat_enabled` | 流式兼容，默认关 |
| `streaming_min_chars` / `streaming_max_chars` | 流式拆段的长度闸门 |

宿主内置分段回复（`platform_settings.segmented_reply.enable`）开启时，插件让位给宿主并记一次 warning：两套切分会把本轮回复拆两遍，条数翻倍。

## 适用范围

- 处理 LLM / Agent 的纯文本结果，链形状须为「若干 At 前缀 + 末尾一条或多条 Plain」。带图片、引用或 At 在中间的链整条跳过，组件保持原位。
- 跳过只有空白或零宽格式字符的片段，@ 跟随第一段可见正文；整条回复都只有这些字符时结束本轮发送。
- 结果已由 `send_message_to_user` 发往当前会话时跳过分段：宿主靠它抑制重复回复，换链后宿主只比对首段。
- 补发通过主动 `send_message` 发送，绕过回复管线的 after-hook 以避免递归；依赖该 hook 的统计插件统计不到补发段。
- 首段发送失败时，从第一段补发整轮内容。
- 补发交接未完成时，整轮内容合并补发。
- 补发中途发送异常时，未送出的段合并后按 1500 字符上限补发。
- 同一会话的多轮补发按序串行。
- 模型超时、解析失败或无可用模型时，按本地标点、围栏与链接规则兜底，边界与代价见 CHANGELOG v0.2.7。

## 流式兼容

默认关闭。开启后按标点与长度实时拆段，配置改动 30 秒内生效，无需重启。

- 插件接管：平台声明支持流式消息时，插件包装 `result.async_stream`，按边界切出各段后交给平台。各平台沿用自身的流式发送方式，WebChat 会把各段追加成一条不断增长的消息。
- 插件跳过：平台声明不支持流式消息时（如 aiocqhttp、Line、Misskey、Slack、Mattermost、企业微信、Discord、微信公众号、Satori），插件跳过包装，回复按平台原有方式发送。

## 许可证

GPL-3.0-or-later
