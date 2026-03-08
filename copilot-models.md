# AI模型配置信息表

> 数据口径：基于 2026-03-08 提供的 GitHub Copilot 模型清单快照整理；相同 `id` 的重复记录已去重；仅保留参考数据可直接验证的字段，不再保留“实测最大输入 Token”这类外部测试值。
>
> 说明：别名 ID 与固定版本 ID 可能共享同一个 `version` 字段，但 `limits`、`picker`、`supported_endpoints` 等元数据仍按各自 `id` 原样记录，例如 `gpt-4o` 与 `gpt-4o-2024-11-20`。
>
> 当前快照共 35 个唯一模型 ID，其中聊天/补全模型 32 个，嵌入模型 3 个；默认聊天模型为 `gpt-5.2`，聊天回退模型为 `gpt-4.1`。主表仅保留当前快照中的模型，上一版存在但当前消失的条目见后文“历史条目”。

## 聊天与补全模型基本信息

| 模型ID | 模型名称 | 厂商 | 类型 | 上下文窗口 | Max Prompt | Max Output | 非流式Max Output | 预览版 | 计费倍率 |
|---|---|---|---|---|---|---|---|---|---|
| claude-opus-4.6 | Claude Opus 4.6 | Anthropic | chat | 144,000 | 128,000 | 64,000 | 16,000 | ❌ | 3 |
| claude-sonnet-4.6 | Claude Sonnet 4.6 | Anthropic | chat | 200,000 | 128,000 | 32,000 | 16,000 | ❌ | 1 |
| gemini-3.1-pro-preview | Gemini 3.1 Pro | Google | chat | 128,000 | 128,000 | 64,000 | - | ✅ | 1 |
| gpt-5.2-codex | GPT-5.2-Codex | OpenAI | chat | 400,000 | 272,000 | 128,000 | - | ❌ | 1 |
| gpt-5.3-codex | GPT-5.3-Codex | OpenAI | chat | 400,000 | 272,000 | 128,000 | - | ❌ | 1 |
| gpt-5.4 | GPT-5.4 | OpenAI | chat | 400,000 | 272,000 | 128,000 | - | ❌ | 1 |
| gpt-5-mini | GPT-5 mini | Azure OpenAI | chat | 264,000 | 128,000 | 64,000 | - | ❌ | 0 |
| gpt-4o-mini-2024-07-18 | GPT-4o mini | Azure OpenAI | chat | 128,000 | 64,000 | 4,096 | - | ❌ | 0 |
| gpt-4o-2024-11-20 | GPT-4o | Azure OpenAI | chat | 128,000 | 64,000 | 16,384 | - | ❌ | 0 |
| gpt-4o-2024-08-06 | GPT-4o | Azure OpenAI | chat | 128,000 | 64,000 | 16,384 | - | ❌ | 0 |
| grok-code-fast-1 | Grok Code Fast 1 | xAI | chat | 128,000 | 128,000 | 64,000 | - | ❌ | 0.25 |
| gpt-5.1 | GPT-5.1 | OpenAI | chat | 264,000 | 128,000 | 64,000 | - | ❌ | 1 |
| gpt-5.1-codex-max | GPT-5.1-Codex-Max | OpenAI | chat | 400,000 | 128,000 | 128,000 | - | ❌ | 1 |
| claude-sonnet-4 | Claude Sonnet 4 | Anthropic | chat | 216,000 | 128,000 | 16,000 | - | ❌ | 1 |
| claude-sonnet-4.5 | Claude Sonnet 4.5 | Anthropic | chat | 144,000 | 128,000 | 32,000 | 16,000 | ❌ | 1 |
| claude-opus-4.5 | Claude Opus 4.5 | Anthropic | chat | 160,000 | 128,000 | 32,000 | 16,000 | ❌ | 3 |
| claude-haiku-4.5 | Claude Haiku 4.5 | Anthropic | chat | 144,000 | 128,000 | 32,000 | 16,000 | ❌ | 0.33 |
| gemini-3-pro-preview | Gemini 3 Pro (Preview) | Google | chat | 128,000 | 128,000 | 64,000 | - | ✅ | 1 |
| gemini-3-flash-preview | Gemini 3 Flash (Preview) | Google | chat | 128,000 | 128,000 | 64,000 | - | ✅ | 0.33 |
| gemini-2.5-pro | Gemini 2.5 Pro | Google | chat | 128,000 | 128,000 | 64,000 | - | ❌ | 1 |
| gpt-4.1-2025-04-14 | GPT-4.1 | Azure OpenAI | chat | 128,000 | 128,000 | 16,384 | - | ❌ | 0 |
| gpt-5.2 | GPT-5.2 | OpenAI | chat | 264,000 | 128,000 | 64,000 | - | ❌ | 1 |
| gpt-41-copilot | GPT-4.1 Copilot | Azure OpenAI | completion | - | - | - | - | ❌ | 0 |
| gpt-3.5-turbo-0613 | GPT 3.5 Turbo | Azure OpenAI | chat | 16,384 | 16,384 | 4,096 | - | ❌ | 0 |
| gpt-4 | GPT 4 | Azure OpenAI | chat | 32,768 | 32,768 | 4,096 | - | ❌ | 0 |
| gpt-4-0613 | GPT 4 | Azure OpenAI | chat | 32,768 | 32,768 | 4,096 | - | ❌ | 0 |
| gpt-4o-2024-05-13 | GPT-4o | Azure OpenAI | chat | 128,000 | 64,000 | 4,096 | - | ❌ | 0 |
| gpt-4-o-preview | GPT-4o | Azure OpenAI | chat | 128,000 | 64,000 | 4,096 | - | ❌ | 0 |
| gpt-4.1 | GPT-4.1 | Azure OpenAI | chat | 128,000 | 128,000 | 16,384 | - | ❌ | 0 |
| gpt-3.5-turbo | GPT 3.5 Turbo | Azure OpenAI | chat | 16,384 | 16,384 | 4,096 | - | ❌ | 0 |
| gpt-4o-mini | GPT-4o mini | Azure OpenAI | chat | 128,000 | 64,000 | 4,096 | - | ❌ | 0 |
| gpt-4o | GPT-4o | Azure OpenAI | chat | 128,000 | 64,000 | 4,096 | - | ❌ | 0 |

## 模型特殊能力支持情况

| 模型ID | vision | tool_calls | parallel_tool_calls | streaming | structured_outputs | reasoning_effort | thinking_budget |
|---|---|---|---|---|---|---|---|
| claude-opus-4.6 | ✅ | ✅ | ✅ | ✅ | ✅ | low/medium/high | adaptive 1,024-32,000 |
| claude-sonnet-4.6 | ✅ | ✅ | ✅ | ✅ | ✅ | low/medium/high | adaptive 1,024-32,000 |
| gemini-3.1-pro-preview | ✅ | ✅ | ✅ | ✅ | ❌ | low/medium/high | 256-32,000 |
| gpt-5.2-codex | ✅ | ✅ | ✅ | ✅ | ✅ | low/medium/high | - |
| gpt-5.3-codex | ✅ | ✅ | ✅ | ✅ | ✅ | low/medium/high | - |
| gpt-5.4 | ✅ | ✅ | ✅ | ✅ | ✅ | low/medium/high | - |
| gpt-5-mini | ✅ | ✅ | ✅ | ✅ | ✅ | low/medium/high | - |
| gpt-4o-mini-2024-07-18 | ❌ | ✅ | ✅ | ✅ | ❌ | - | - |
| gpt-4o-2024-11-20 | ✅ | ✅ | ✅ | ✅ | ❌ | - | - |
| gpt-4o-2024-08-06 | ❌ | ✅ | ✅ | ✅ | ❌ | - | - |
| grok-code-fast-1 | ❌ | ✅ | ❌ | ✅ | ✅ | - | - |
| gpt-5.1 | ✅ | ✅ | ✅ | ✅ | ✅ | none/low/medium/high | - |
| gpt-5.1-codex-max | ✅ | ✅ | ✅ | ✅ | ✅ | low/medium/high | - |
| claude-sonnet-4 | ✅ | ✅ | ✅ | ✅ | ❌ | - | 1,024-32,000 |
| claude-sonnet-4.5 | ✅ | ✅ | ✅ | ✅ | ❌ | - | 1,024-32,000 |
| claude-opus-4.5 | ✅ | ✅ | ✅ | ✅ | ❌ | - | 1,024-32,000 |
| claude-haiku-4.5 | ✅ | ✅ | ✅ | ✅ | ❌ | - | 1,024-32,000 |
| gemini-3-pro-preview | ✅ | ✅ | ✅ | ✅ | ❌ | low/high | 256-32,000 |
| gemini-3-flash-preview | ✅ | ✅ | ✅ | ✅ | ❌ | low/medium/high | 256-32,000 |
| gemini-2.5-pro | ✅ | ✅ | ✅ | ✅ | ❌ | - | 128-32,768 |
| gpt-4.1-2025-04-14 | ✅ | ✅ | ✅ | ✅ | ✅ | - | - |
| gpt-5.2 | ✅ | ✅ | ✅ | ✅ | ✅ | - | - |
| gpt-41-copilot | ❌ | ❌ | ❌ | ✅ | ❌ | - | - |
| gpt-3.5-turbo-0613 | ❌ | ✅ | ❌ | ✅ | ❌ | - | - |
| gpt-4 | ❌ | ✅ | ❌ | ✅ | ❌ | - | - |
| gpt-4-0613 | ❌ | ✅ | ❌ | ✅ | ❌ | - | - |
| gpt-4o-2024-05-13 | ✅ | ✅ | ✅ | ✅ | ❌ | - | - |
| gpt-4-o-preview | ❌ | ✅ | ✅ | ✅ | ❌ | - | - |
| gpt-4.1 | ✅ | ✅ | ✅ | ✅ | ✅ | - | - |
| gpt-3.5-turbo | ❌ | ✅ | ❌ | ✅ | ❌ | - | - |
| gpt-4o-mini | ❌ | ✅ | ✅ | ✅ | ❌ | - | - |
| gpt-4o | ✅ | ✅ | ✅ | ✅ | ❌ | - | - |

## 模型选择器与访问限制

| 模型ID | Picker分类 | Picker可见 | 默认聊天 | 回退聊天 | 支持端点 | 计划限制 |
|---|---|---|---|---|---|---|
| claude-opus-4.6 | powerful | ✅ | ❌ | ❌ | /v1/messages / /chat/completions | pro/edu/pro_plus/business/enterprise |
| claude-sonnet-4.6 | versatile | ✅ | ❌ | ❌ | /chat/completions / /v1/messages | pro/edu/pro_plus/business/enterprise |
| gemini-3.1-pro-preview | powerful | ✅ | ❌ | ❌ | /chat/completions | pro/pro_plus/business/enterprise |
| gpt-5.2-codex | powerful | ✅ | ❌ | ❌ | /responses | pro/edu/pro_plus/business/enterprise |
| gpt-5.3-codex | powerful | ✅ | ❌ | ❌ | /responses | pro/edu/pro_plus/business/enterprise |
| gpt-5.4 | powerful | ✅ | ❌ | ❌ | /responses | pro/edu/pro_plus/business/enterprise |
| gpt-5-mini | lightweight | ✅ | ❌ | ❌ | /chat/completions / /responses | - |
| grok-code-fast-1 | lightweight | ✅ | ❌ | ❌ | - | - |
| gpt-5.1 | versatile | ✅ | ❌ | ❌ | /chat/completions / /responses | pro/pro_plus/max/business/enterprise/edu |
| gpt-5.1-codex-max | powerful | ✅ | ❌ | ❌ | /responses | pro/pro_plus/max/business/enterprise/edu |
| claude-sonnet-4 | versatile | ✅ | ❌ | ❌ | /chat/completions / /v1/messages | pro/pro_plus/max/business/enterprise/edu |
| claude-sonnet-4.5 | versatile | ✅ | ❌ | ❌ | /chat/completions / /v1/messages | pro/pro_plus/max/business/enterprise/edu |
| claude-opus-4.5 | powerful | ✅ | ❌ | ❌ | /chat/completions / /v1/messages | pro/pro_plus/max/business/enterprise/edu |
| claude-haiku-4.5 | versatile | ✅ | ❌ | ❌ | /chat/completions / /v1/messages | - |
| gemini-3-pro-preview | powerful | ✅ | ❌ | ❌ | - | pro/pro_plus/max/business/enterprise/edu |
| gemini-3-flash-preview | lightweight | ✅ | ❌ | ❌ | - | pro/pro_plus/max/business/enterprise/edu |
| gemini-2.5-pro | powerful | ✅ | ❌ | ❌ | - | pro/pro_plus/max/business/enterprise/edu |
| gpt-5.2 | versatile | ✅ | ✅ | ❌ | /chat/completions / /responses | pro/pro_plus/max/business/enterprise/edu |
| gpt-41-copilot | versatile | ✅ | ❌ | ❌ | - | - |
| gpt-4.1 | versatile | ✅ | ❌ | ✅ | - | pro/pro_plus/max/business/enterprise/edu |
| gpt-4o | versatile | ✅ | ❌ | ❌ | - | - |

## 历史条目（当前快照未出现）

> 以下模型来自上一版文档，为保留变更痕迹继续展示；它们未出现在 2026-03-08 快照中，因此按历史条目处理，并使用删除线标记。除非 GitHub 官方另有说明，这里不直接表述为“官方下架”。

| 模型ID | 模型名称 | 厂商 | 版本 | 上次记录上下文窗口 | 上次记录最大输出 | 备注 |
|---|---|---|---|---|---|---|
| ~~gpt-5~~ | ~~GPT-5 (Preview)~~ | ~~Azure OpenAI~~ | ~~gpt-5~~ | ~~128,000~~ | ~~64,000~~ | 上一版文档中存在，但未出现在 2026-03-08 快照中 |
| ~~o3-mini~~ | ~~o3-mini~~ | ~~Azure OpenAI~~ | ~~o3-mini-2025-01-31~~ | ~~200,000~~ | ~~100,000~~ | 上一版文档中存在，但未出现在 2026-03-08 快照中 |
| ~~o3-mini-2025-01-31~~ | ~~o3-mini~~ | ~~Azure OpenAI~~ | ~~o3-mini-2025-01-31~~ | ~~200,000~~ | ~~100,000~~ | 上一版文档中存在，但未出现在 2026-03-08 快照中 |
| ~~o3-mini-paygo~~ | ~~o3-mini~~ | ~~Azure OpenAI~~ | ~~o3-mini-paygo~~ | ~~200,000~~ | ~~100,000~~ | 上一版文档中存在，但未出现在 2026-03-08 快照中 |
| ~~gpt-4o-copilot~~ | ~~GPT-4o Copilot~~ | ~~Azure OpenAI~~ | ~~gpt-4o-copilot~~ | ~~-~~ | ~~-~~ | 上一版文档中存在，但未出现在 2026-03-08 快照中 |
| ~~claude-3.5-sonnet~~ | ~~Claude Sonnet 3.5~~ | ~~Anthropic~~ | ~~claude-3.5-sonnet~~ | ~~90,000~~ | ~~8,192~~ | 上一版文档中存在，但未出现在 2026-03-08 快照中 |
| ~~claude-3.7-sonnet~~ | ~~Claude Sonnet 3.7~~ | ~~Anthropic~~ | ~~claude-3.7-sonnet~~ | ~~200,000~~ | ~~16,384~~ | 上一版文档中存在，但未出现在 2026-03-08 快照中 |
| ~~claude-3.7-sonnet-thought~~ | ~~Claude Sonnet 3.7 Thinking~~ | ~~Anthropic~~ | ~~claude-3.7-sonnet-thought~~ | ~~200,000~~ | ~~16,384~~ | 上一版文档中存在，但未出现在 2026-03-08 快照中 |
| ~~gemini-2.0-flash-001~~ | ~~Gemini 2.0 Flash~~ | ~~Google~~ | ~~gemini-2.0-flash-001~~ | ~~1,000,000~~ | ~~8,192~~ | 上一版文档中存在，但未出现在 2026-03-08 快照中 |
| ~~o4-mini~~ | ~~o4-mini (Preview)~~ | ~~Azure OpenAI~~ | ~~o4-mini-2025-04-16~~ | ~~128,000~~ | ~~16,384~~ | 上一版文档中存在，但未出现在 2026-03-08 快照中 |
| ~~o4-mini-2025-04-16~~ | ~~o4-mini (Preview)~~ | ~~OpenAI~~ | ~~o4-mini-2025-04-16~~ | ~~200,000~~ | ~~100,000~~ | 上一版文档中存在，但未出现在 2026-03-08 快照中 |

## 嵌入模型

| 模型ID | 模型名称 | 厂商 | 版本 | 最大输入数 | 支持自定义维度 | Tokenizer |
|---|---|---|---|---|---|---|
| text-embedding-3-small | Embedding V3 small | Azure OpenAI | text-embedding-3-small | 512 | ✅ | cl100k_base |
| text-embedding-3-small-inference | Embedding V3 small (Inference) | Azure OpenAI | text-embedding-3-small | - | ✅ | cl100k_base |
| text-embedding-ada-002 | Embedding V2 Ada | Azure OpenAI | text-embedding-3-small | 512 | ❌ | cl100k_base |

> 注：`text-embedding-ada-002` 的 `version` 字段在参考快照中仍为 `text-embedding-3-small`，这里按原始数据保留，不做推断修正。

## 模型Tokenizer信息

| 模型类别 | 使用的Tokenizer |
|---|---|
| GPT-3.5 / GPT-4 / Embedding V2 | cl100k_base |
| GPT-4o / GPT-4.1 / GPT-5 / Claude / Gemini / Grok | o200k_base |
| Embedding V3 | cl100k_base |
