# CHANGELOG

## v0.2.3 (2026-10-15)

### Bug Fixes
- 修复本地兜底拆分丢弃段首句读符的内容丢失缺陷：`。？！~…!?` 位于段首时无法被
  `SENTENCE_BOUNDARY_RE` 匹配（匹配要求以非句读符开头）而被静默丢弃。改为先剥离
  段首句读符再拼回首句，段首内容不再丢失
- 修复 `terminate()` 落在分段调用期间时回复被截断的缺陷：`on_decorating_result`
  缺少停止检查，替换结果链后补发又被 `after_message_sent` 丢弃，用户只看到首段
- 修复补发去重按 `id(result)` 判定的缺陷：result 释放后地址可能被同一 event 上的
  新 result 复用（agent 多步循环每步都新建 `MessageEventResult`），导致新回复被
  误判为已处理而静默跳过分段。改为弱引用身份标记
- 修复流式兼容把 Live Mode 音频链当聊天文本切碎的缺陷：`audio_chunk` 链的
  base64 音频被按 `streaming_max_chars` 硬切成多条文本垃圾，且链类型丢失导致
  前端收不到音频。平台专用链（`audio_chunk`/`reasoning`/`tool_call` 等）现原样
  透传，不再拆分
- 修复流式换行边界绕过 `min_chars`/`max_chars` 的缺陷：换行曾在任何长度判断之前
  直接断段，模型按行输出（列表、分步骤回复）时被拆成大量 3~5 字碎消息，超过
  `max_chars` 的换行也会让消息等太久。换行现与句读符走同一套长度闸门，第一个
  换行不满足时继续寻找后续满足条件的换行
- 修复 astrbot 4.16–4.19 上模型结果守卫静默失效的缺陷：`is_model_result` 自
  4.20 才存在，缺失时任何插件的纯文本结果都会被误当成模型回复拿去分段（与
  README 声明的行为边界不符）。缺失时回退 `is_llm_result`
- 修复流式分段丢失边界换行符的缺陷：边界取换行符之前且 `strip()` 剥掉尾部
  空白，而所有平台都把各段 Plain 无分隔符拼接——markdown 列表、分步说明会被
  黏成一行。边界现取换行 run 末尾（含换行符），分段只剥空格/制表符
- 修复流式兼容吞掉源流异常的缺陷：异常被吞后平台以为流正常结束，用户拿到
  静默截断的回复还以为发送成功。现先 flush 已缓冲内容再上抛，与未打补丁的
  核心语义一致
- 修复流式强句读符分支绕过 `max_chars` 的缺陷：超长句子会整段穿过上限
  （违背 `streaming_max_chars`"超过该长度时按弱边界/换行切分"的对外契约），
  且同一文本的分段结果随 provider 的 chunk 大小漂移（chunk=1 切 [90,6]、
  chunk=200 切整段）。强句读符现与换行走同一套 min/max 双闸门
- 修复 `find_stream_boundary` 全文扫描的二次复杂度：单次消费 N 字符 chunk 的
  成本与切分段长成反比的二次复杂度（约 O(N²/max_chars)，`max_chars` 调小时更
  差），10 万字符单 chunk 阻塞事件循环数秒。改为只在 `[min_chars-1, max_chars)`
  窗口内定位边界（合格边界的起点必落在窗口内）：边界查找耗时与输入长度无关，
  强/弱句读符分支经开发期差分验证行为不变，上述场景降到毫秒级
- 修复模型输出中右括号过剩把后续所有段粘成一条的缺陷：`has_unbalanced_brackets`
  按个数比较判定，孤立的右括号（或 `:)` 类表情）会使 `merge_segments_balancing_brackets`
  无限吸收后续段。改为仅左括号过剩（含未闭合）才算不均衡
- 修复配置含 `inf`（JSON `1e999`/`Infinity`）时插件加载失败的缺陷：`as_int(inf)`
  抛 `OverflowError` 未被捕获，构造期抛出后每条消息的 hook 都抛异常。`as_int`/
  `as_float` 现捕获 `OverflowError`
- 修复模型把数组包在对象里返回时整份结果作废的缺陷：`extract_json_array_text`
  取第一个"能解析成 JSON"的候选而不要求是数组。现优先取解析结果为 list 的候选
- 修复深层嵌套 JSON 抛 `RecursionError` 而非 `SegmentationParseError` 的契约违背：
  `parse_segments_from_model_output` 现收敛该异常（此前被通用兜底捕获，日志打成
  "LLM 调用失败"+完整 traceback）
- 修复平台发送结果 `retcode` 为字符串 `"0"` 时被误判为发送失败的缺陷：误判会
  中止全部剩余补发，导致用户丢失消息内容
- 修复 `strip_thinking_content` 在大量未闭合 `<think>` 开标签下的二次复杂度回溯
  （病态输入秒级、阻塞事件循环）：无闭合标签时先廉价探测再决定是否执行正则
  替换（语义等价，病态输入从秒级降到毫秒级）

### Refactoring
- `_stage_sentences` 段首句读符剥离改用 `lstrip`，并消除每个匹配两次的
  `_strip_single_trailing_period` 重复计算

### Testing
- 新增段首句读符保留的单测与金样（含括号拆分后段首出现句读符的场景）
- 新增停止中跳过分段、去重身份标记（弱引用路径 + 地址复用红灯）回归测试
- 新增平台专用链透传、换行长度闸门（含"寻找后续满足条件的换行"）、右括号过剩
  合并、`inf` 配置、数组候选优先、深层嵌套解析、`retcode` 字符串零值、
  `llm_generate` 旧签名降级的回归测试与金样
- 新增 `is_llm_result` 回退（astrbot 4.16–4.19 守卫）、break flush 非空 buffer、
  `should_segment_text` 动作分支（输入长于 min_length）的测试
- 新增边界换行符保留（`find_stream_boundary` 含换行、`pop_stream_segments`
  只剥空格）、源流异常 flush 后上抛、多 Plain chunk 累加、非 Plain/平台链前
  flush、强句读符受 max_chars 约束（不再产生超长分段）的测试
- 新增变异审计发现的契约缺口守卫：跨段括号配平与尾 buffer flush、四对括号
  配平、纯动作段判定（闭括号在段中）、`normalize_segments` 标量/空段契约、
  配置下钳制（负值）、`as_bool` 真假词表、非法浮点回落、LLM 抛异常兜底、
  空 completion、配置 provider 优先权、`is_model_result` 优先序、clear_extras
  后去重安全、补发期间持会话守卫、terminate 通知子系统
- 新增 `calculate_send_delay` 四个契约的精确断言（零抖动精确值、抖动区间、
  每字权重、rng 注入）、每个强句读符的句中边界参数化用例、四对括号的
  `split_text_at_brackets` 参数化用例
- 新增补发生命线中止契约（`send_message` 返回 False、补发中被禁用）、
  `test_terminate_clean` 改为先制造在飞状态再终止（原断言空 vs 空）
- 删除 3 个被金样/兄弟用例完全覆盖的 smoke test（`test_plain_chain_pure`、
  `test_plain_array`、`test_no_op`）
- `test_streaming_result_passthrough` 改用真实 `ResultContentType` 枚举并断言
  fake 确为流式结果（原字符串常量使 `is_streaming_result` 恒假，测试近恒真）
- 补充 `brief_exception` dict-result 截断覆盖；`test_fenced_json` 输入加入尾部
  括号数组，使 ```json fence 分支可被烧红
- 删除 `_CONFIG_KEYS` 字面量快照测试（dataclass 增加字段即误报）与手工构造的
  补丁装卸测试（真实安装/卸载已由 test_plugin_load 覆盖）
- `calculate_send_delay` 上限钳制测试从 test_follow_up.py 移至 test_segmentation.py

### Chore
- 移除 CHANGELOG 中指向 docs/manual-verify.md 的引用（该文件已删除，v0.2.1 历史条目加注记说明）

### Documentation
- README 行为边界补充流式预分段的平台生效范围（逐链发送的平台才生效）
- `_conf_schema.json` 的 `streaming_min_chars`/`streaming_max_chars` hint 补充
  换行边界同样受长度闸门约束

## v0.2.2 (2026-08-07)

### Bug Fixes
- 修复运行时切换 `streaming_compat_enabled` 在纯流式流量下不生效的缺陷（D-1）。
  根因：流式结果在 `DecorateStage.process` 与 `RespondStage.process` 均提前返回，
  两个 hook 都不触发，补丁装卸失去驱动。改为 30 秒周期同步（`_sync_loop`）。
- 修复测试对全局 `asyncio.sleep` 的 monkeypatch 污染（D-1 暴露）：改 patch
  `calculate_send_delay`，避免影响常驻协程的时间语义。

### Refactoring
- 折叠 `streaming/patches.py` 中单目标补丁的未使用泛化（PatchHandle 单实例
  脚手架，净减 32 行；7 步状态机差分验证等价）
- 流式弱边界取窗口内最晚位置（O-13 B：段落更长、碎片更少）

### Testing
- 新增真实 Stage 流式集成测试（O-11）：把 O-1/A4 的一次性探针固化为常驻
  守卫（提前返回属实 / 补丁包装 / 卸载原样）
- 新增 `astrbot_version` 上界断言（O-12）：锁定 A4 的成立前提 <5
- 新增 D-1 周期同步测试（不改配置不发消息也能装卸补丁 / terminate 回收任务）
- 手工验证清单扩充至 14 项含部署步骤（真实 QQ 环境执行）

## v0.2.1 (2026-08-07)

### Testing & Quality
- 新增 schema 默认值对齐守卫（`_conf_schema.json` default ↔ dataclass default）
- 新增零运行时依赖 import 边界守卫（相对导入按 AST `level` 判定）
- 金样扩充至 17+13（从 9+9）：未闭合括号、嵌套括号、纯标点、CRLF、
  未闭合 thinking、emoji 边界、长文本 cap、尾随垃圾、无标签 fence 含代码、
  数组内换行、英文括号等
- 单元测试补充：`extract_json_array_text` 无括号、`cap_segments` max=1、
  嵌套括号合并、流式边界（换行在 0 位/弱边界不足 min/无边界硬切）

### Refactoring
- `_CONFIG_KEYS` 从 dataclass 字段自动派生（消除 14 行手工维护）
- 删除 `_stage_lines` 中不可达兜底（差分 4000 例实证等价）
- Jitter 常量收敛至 `bounds.py`（`DELAY_JITTER_SECONDS`）
- shutdown 取消日志降为 DEBUG（高频预期路径降噪）
- 删除 `on_decorating_result` 中不可达流式分支（`ResultDecorateStage` 对
  `STREAMING_RESULT` 提前返回；流式兼容仅由补丁路径承担，`metadata.yaml`
  锁定 `<5`）

### Chore
- 删除无引用 `data/` 目录（cmd_config.json + t2i_templates/，-1313 行）

### Documentation
- requirements-dev.txt 补充 pytest-asyncio 与 astrbot 依赖
- 新增 docs/manual-verify.md 发布前手工验证清单（该文件后续已移除）

## v0.2.0

- 拆分模块：settings / segmenter / follow_up / chain_utils / streaming / errors
- 配置上下限单一来源（`bounds.py` 对齐 `_conf_schema.json`）；streaming min/max 互斥校正
- `SettingsLoader` 按配置值指纹缓存，hook 热路径不再全量重解析
- 移除旧版兼容：`get_using_provider` 回退、位置参数探测（锁定 AstrBot `>=4.16,<5`）
- 流式 C-light：默认不 import/不装补丁；仅 `ResultDecorateStage`；sync 幂等短路
- 修复流式混合链组件顺序（At/Plain 保持原始顺序）；删死代码与误导注释
- follow-up 簿记（去重/pending/清态）收进 `FollowUpDispatcher`
- pytest 95 项：纯逻辑、settings 缓存、boundary、segmenter（超时/取消/无 provider）、
  follow_up（失败停止/guard）、chain_utils、stream wrapper、patch 多场景、
  端到端流程、金样
