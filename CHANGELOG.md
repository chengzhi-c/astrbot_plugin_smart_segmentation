# CHANGELOG

## v0.2.4 (2026-10-16)
- 修复：流式分段的换行 / 长度闸门 / emoji 字形簇边界；thinking 病态输入回溯、
  异名标签误删正文、孤立 `\r` 归一化；补发中途失败合并补发、同会话串行、TTL
  认领、停止/禁用兜底；流式补丁装饰异常隔离、归属校验后重装、`terminate` 卸载；
  模型输出末尾 `[` 兜底、`as_bool` 认 1/0 与浮点、配置异常降级
- 重构：删除 `segmenter` 的 provider 能力探测层、`settings` 的指纹缓存与死分支
- 测试：补齐回归与金样，修正假守卫，删除恒真与只锁旧签名的用例

## v0.2.3 (2026-10-15)
- 修复：流式换行/强句读符统一受 min/max 闸门约束、保留边界换行符、源流异常先
  flush 再上抛、平台专用链原样透传、边界查找改窗口内（消除二次复杂度）；本地
  兜底保留段首句读符、右括号过剩不再粘连、thinking 未闭合开标签不再二次回溯；
  `terminate` 落在分段期间不再截断、补发去重改按对象身份、`is_model_result` 缺失
  时回退 `is_llm_result`、`inf` 配置不再加载失败、数组包对象仍可取到、深层嵌套
  收敛为解析错误、`retcode` 字符串 `"0"` 不再误判
- 测试：补齐上述回归与金样，删除被完全覆盖的 smoke 测试

## v0.2.2 (2026-08-07)
- 修复：运行时切换 `streaming_compat_enabled` 在纯流式流量下不生效（改为 30 秒
  周期同步）
- 重构：折叠 `streaming/patches.py` 的单目标补丁脚手架；流式弱边界取窗口内最晚位置
- 测试：真实 Stage 流式集成守卫、`astrbot_version` 上界断言、周期同步测试

## v0.2.1 (2026-08-07)
- 测试：金样扩充至 17+13；schema 默认值对齐、零运行时依赖 import 边界守卫
- 重构：`_CONFIG_KEYS` 自动派生、jitter 常量收敛、删除不可达分支；删除无引用
  `data/` 目录

## v0.2.0
- 拆分模块（settings / segmenter / follow_up / chain_utils / streaming / errors）
- 配置上下限单一来源（`bounds.py`）；移除旧版兼容，锁定 `>=4.16,<5`
- 流式 C-light：默认不 import/不装补丁，仅 patch `ResultDecorateStage`
- follow-up 簿记收进 `FollowUpDispatcher`；修复流式混合链组件顺序
