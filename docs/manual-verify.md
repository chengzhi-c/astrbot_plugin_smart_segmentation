# 手工验证清单（v0.2.2 发布前）

执行环境：astrbot 4.23.3，本地 QQ 测试群。
执行人：____　日期：____

## 0. 部署步骤（v0.2.2）

```powershell
# 1. 备份活动副本（务必先做）
$active = "E:\项目\astrbot_plugin_smart_segmentation"
$new    = "E:\项目\项目备份\astrbot_plugin_smart_segmentation-main"
Copy-Item $active "$active.bak-$(Get-Date -Format yyyyMMdd-HHmmss)" -Recurse

# 2. 清理活动副本内容（保留目录本身与 data/ 运行时目录）
Get-ChildItem $active -Exclude "data" | Remove-Item -Recurse -Force

# 3. 复制新版（排除 git/缓存/测试产物）
Copy-Item "$new\*" $active -Recurse -Exclude ".git", "__pycache__", ".pytest_cache", "data", "docs"
```

AstrBot 侧前置：
- `platform_settings.segmented_reply.enable = false`（避免与插件重复分段）
- 确认 LLM provider 可用；日志级别 INFO 或 DEBUG
- 插件配置初始值：`enabled: true`、`streaming_compat_enabled: false`、`min_length: 12`、`max_segments: 5`、`timeout_seconds: 30`、`delay_base: 0.8`、`delay_per_char: 0.025`、`delay_max: 2.5`
- 重启 AstrBot 使插件生效

日志查看（AstrBot 日志目录，如 `data/astrbot.log` 或 WebUI 控制台）：
```powershell
# 过滤插件相关日志
Select-String -Path <astrbot.log> -Pattern "智能分段"
```

## 1. 验证矩阵（14 项）

| # | 场景 | 配置 | 操作 | 预期 |
|---|------|------|------|------|
| 1 | 非流式 × compat 关 | compat=false，provider 非流式 | 发一个长回复提问 | 首段立即到、剩余段有可见间隔逐条到；日志「发送前处理完成，共 N 段」；无「补丁已启用」 |
| 2 | 非流式 × compat 开 | compat=true | 同上 | 与 #1 完全一致（compat 不影响非流式） |
| 3 | 流式 × compat 关 | provider 开流式，compat=false | 同上 | 原样流式（打字机效果，无分条）；日志无「补丁已启用」「已接管流式输出」 |
| 4 | 流式 × compat 开 | 流式 + compat=true | 同上 | 按标点/长度分条；日志有「补丁已启用」+「已接管流式输出」 |
| 5 | 运行时开启 compat（D-1 重点） | 从 #3 状态出发，纯流式流量 | 改 compat false→true，不发非流式消息，等 35 秒后发流式提问 | 分条生效；日志出现「补丁已启用」。修复前此项失败 |
| 6 | 运行时关闭 compat（D-1 重点） | 从 #4 状态出发 | 改 compat true→false，等 35 秒后发流式提问 | 回到原样流式，不分条 |
| 7 | 停用插件 | #4 状态 | WebUI 停用插件 | 流式还原原样；无残留补丁 WARNING |
| 8 | 热重载 | #4 状态 | 重载插件 | 无重复包装（一条消息不被切两遍）；日志无「检测到旧实例残留」以外的异常 |
| 9 | 分段模型超时 | timeout_seconds=0.1 | 发提问 | 本地规则兜底分段，内容不丢；日志「调用超时…改用本地规则兜底」 |
| 10 | provider 不可用 | provider_id 填不存在的 id | 发提问 | 本地规则兜底；日志「LLM 调用失败」或「未找到可用 provider_id」 |
| 11 | 发送中断 | 正常配置 | 补发过程中断网 / 踢出群 | 剩余补发停止；日志 WARNING「补发发送失败，已停止剩余补发」；无未捕获堆栈 |
| 12 | action-only 文本 | 正常配置 | 诱导 LLM 只回（沉默）类纯括号内容 | 不分段、不被拆成（沉默 + ） |
| 13 | thinking 标签 | 会输出 think 标签的模型 | 发提问 | 标签及内容被剥离，用户侧看不到 |
| 14 | max_segments 收敛 | max_segments=2 | 发会得到 5 段以上的提问 | 恰好 2 条，第 2 条为剩余内容合并 |

## 2. 每项执行模板

### #5 运行时开启 compat（D-1 重点）

- **配置前**：`streaming_compat_enabled: false`，provider 流式开启
- **操作**：WebUI 改 compat 为 true，保存。不发任何消息，等待 35 秒。发送「介绍一下光合作用」
- **观察**：
  - 输出形态：（分条 / 原样流式）
  - 日志「智能分段流式兼容补丁已启用」：（出现时间戳 / 未出现）
  - 日志「智能分段已接管流式输出」：（出现 / 未出现）
- **结论**：✅ / ❌
- **备注**：

### #6 运行时关闭 compat（D-1 重点）

- **配置前**：`streaming_compat_enabled: true`，流式分条生效中
- **操作**：改 compat 为 false，保存。不发任何消息，等待 35 秒。发送「介绍一下光合作用」
- **观察**：
  - 输出形态：（分条 / 原样流式）
  - 日志「补丁已启用」之后是否出现新的（不应出现）
- **结论**：✅ / ❌
- **备注**：

### 其余 12 项

按矩阵表逐项执行，每项记录：配置快照、操作、观察到的实际现象、日志关键行、结论。任一项不符预期即停止，记录现象后回到代码层排查——不允许「大概是环境问题」这类结论收尾。

## 3. 红灯处理协议

1. 停止后续项
2. 保存日志片段与配置快照
3. 回到代码层定位（本仓库 git log 可回退到任一 commit）
4. 修复后重跑该项与受影响的相邻项
