# AI模型配置信息表

> 数据口径：基于 `local/model.json` 当前快照整理。
>
> 去重规则：原始数据共 37 条记录，按 `id` 去重后为 35 个唯一模型 ID；`gpt-4` 与 `gpt-4-o-preview` 在源数据中各重复出现 2 次，本文仅保留 1 条。
>
> 模型范围：生成类模型 32 个（chat 31 + completion 1)，嵌入模型 3 个；Picker 可见模型 21 个，非 Picker 生成模型 11 个。
>
> 默认聊天模型：`gpt-5.2`；聊天回退模型：`gpt-4.1`。
>
> 重要说明：`id` 与 `version` 相同并不意味着所有元数据一致。别名 ID 与固定版本 ID 可能共享同一个 `version`，但 `limits`、`billing`、`supported_endpoints`、`capabilities.supports` 仍按各自 `id` 独立记录。
>
> 本次刷新相较上一版的结构调整：
> 1. 把 Picker 可见模型与非 Picker 生成模型拆表。
> 2. 新增“别名与固定版本对照”表，显式展示 `id -> version`。
> 3. 能力表新增 `Max Images`、`图像格式`、`adaptive_thinking` 三列。
> 4. 访问表移除冗余的 `Picker可见` 列，新增 `Premium` 列。
> 5. `支持端点` 列中的 `-` 表示当前快照未声明该字段，不等同于“完全不可用”。

## 一、别名与固定版本对照

| 模型ID | 实际 version | family | 类型 | Picker | 备注 |
|---|---|---|---|---|---|
| text-embedding-3-small-inference | text-embedding-3-small | text-embedding-3-small | embeddings | ❌ | 推理别名，version 仍指向 text-embedding-3-small |
| gpt-4 | gpt-4-0613 | gpt-4 | chat | ❌ | 旧版别名；源数据中该 ID 有重复记录 |
| gpt-4-o-preview | gpt-4o-2024-05-13 | gpt-4o | chat | ❌ | preview 别名；其 vision 能力与同 version 固定版本条目不一致 |
| gpt-4.1 | gpt-4.1-2025-04-14 | gpt-4.1 | chat | ✅ | picker 别名，同时是聊天回退模型 |
| gpt-3.5-turbo | gpt-3.5-turbo-0613 | gpt-3.5-turbo | chat | ❌ | 旧版别名；billing 中保留了 restricted_to |
| gpt-4o-mini | gpt-4o-mini-2024-07-18 | gpt-4o-mini | chat | ❌ | 固定版本别名 |
| gpt-4o | gpt-4o-2024-11-20 | gpt-4o | chat | ✅ | picker 别名，指向当前 GPT-4o 固定版本 |
| text-embedding-ada-002 | text-embedding-3-small | text-embedding-ada-002 | embeddings | ❌ | 源数据中 version 仍为 text-embedding-3-small，按原样保留 |

## 二、Picker 模型：基本信息

| 模型ID | version | 模型名称 | 厂商 | 类型 | 上下文窗口 | Max Prompt | Max Output | 非流式 Max Output | 预览版 | 计费倍率 |
|---|---|---|---|---|---|---|---|---|---|---|
| claude-opus-4.6 | claude-opus-4.6 | Claude Opus 4.6 | Anthropic | chat | 144,000 | 128,000 | 64,000 | 16,000 | ❌ | 3 |
| claude-sonnet-4.6 | claude-sonnet-4.6 | Claude Sonnet 4.6 | Anthropic | chat | 200,000 | 128,000 | 32,000 | 16,000 | ❌ | 1 |
| gemini-3.1-pro-preview | gemini-3.1-pro-preview | Gemini 3.1 Pro | Google | chat | 128,000 | 128,000 | 64,000 | - | ✅ | 1 |
| gpt-5.2-codex | gpt-5.2-codex | GPT-5.2-Codex | OpenAI | chat | 400,000 | 272,000 | 128,000 | - | ❌ | 1 |
| gpt-5.3-codex | gpt-5.3-codex | GPT-5.3-Codex | OpenAI | chat | 400,000 | 272,000 | 128,000 | - | ❌ | 1 |
| gpt-5.4 | gpt-5.4 | GPT-5.4 | OpenAI | chat | 400,000 | 272,000 | 128,000 | - | ❌ | 1 |
| gpt-5-mini | gpt-5-mini | GPT-5 mini | Azure OpenAI | chat | 264,000 | 128,000 | 64,000 | - | ❌ | 0 |
| grok-code-fast-1 | grok-code-fast-1 | Grok Code Fast 1 | xAI | chat | 128,000 | 128,000 | 64,000 | - | ❌ | 0.25 |
| gpt-5.1 | gpt-5.1 | GPT-5.1 | OpenAI | chat | 264,000 | 128,000 | 64,000 | - | ❌ | 1 |
| gpt-5.1-codex-max | gpt-5.1-codex-max | GPT-5.1-Codex-Max | OpenAI | chat | 400,000 | 128,000 | 128,000 | - | ❌ | 1 |
| claude-sonnet-4 | claude-sonnet-4 | Claude Sonnet 4 | Anthropic | chat | 216,000 | 128,000 | 16,000 | - | ❌ | 1 |
| claude-sonnet-4.5 | claude-sonnet-4.5 | Claude Sonnet 4.5 | Anthropic | chat | 144,000 | 128,000 | 32,000 | 16,000 | ❌ | 1 |
| claude-opus-4.5 | claude-opus-4.5 | Claude Opus 4.5 | Anthropic | chat | 160,000 | 128,000 | 32,000 | 16,000 | ❌ | 3 |
| claude-haiku-4.5 | claude-haiku-4.5 | Claude Haiku 4.5 | Anthropic | chat | 144,000 | 128,000 | 32,000 | 16,000 | ❌ | 0.33 |
| gemini-3-pro-preview | gemini-3-pro-preview | Gemini 3 Pro (Preview) | Google | chat | 128,000 | 128,000 | 64,000 | - | ✅ | 1 |
| gemini-3-flash-preview | gemini-3-flash-preview | Gemini 3 Flash (Preview) | Google | chat | 128,000 | 128,000 | 64,000 | - | ✅ | 0.33 |
| gemini-2.5-pro | gemini-2.5-pro | Gemini 2.5 Pro | Google | chat | 128,000 | 128,000 | 64,000 | - | ❌ | 1 |
| gpt-5.2 | gpt-5.2 | GPT-5.2 | OpenAI | chat | 264,000 | 128,000 | 64,000 | - | ❌ | 1 |
| gpt-41-copilot | gpt-41-copilot | GPT-4.1 Copilot | Azure OpenAI | completion | - | - | - | - | ❌ | 0 |
| gpt-4.1 | gpt-4.1-2025-04-14 | GPT-4.1 | Azure OpenAI | chat | 128,000 | 128,000 | 16,384 | - | ❌ | 0 |
| gpt-4o | gpt-4o-2024-11-20 | GPT-4o | Azure OpenAI | chat | 128,000 | 64,000 | 4,096 | - | ❌ | 0 |

## 三、Picker 模型：能力与输入限制

| 模型ID | vision | Max Images | 图像格式 | tool_calls | parallel_tool_calls | streaming | structured_outputs | reasoning_effort | adaptive_thinking | thinking_budget |
|---|---|---|---|---|---|---|---|---|---|---|
| claude-opus-4.6 | ✅ | 1 | jpeg/png/webp | ✅ | ✅ | ✅ | ✅ | low/medium/high | ✅ | 1,024-32,000 |
| claude-sonnet-4.6 | ✅ | 5 | jpeg/png/webp | ✅ | ✅ | ✅ | ✅ | low/medium/high | ✅ | 1,024-32,000 |
| gemini-3.1-pro-preview | ✅ | 10 | jpeg/png/webp | ✅ | ✅ | ✅ | ❌ | low/medium/high | ❌ | 256-32,000 |
| gpt-5.2-codex | ✅ | 1 | jpeg/png/webp/gif | ✅ | ✅ | ✅ | ✅ | low/medium/high | ❌ | - |
| gpt-5.3-codex | ✅ | 1 | jpeg/png/webp/gif | ✅ | ✅ | ✅ | ✅ | low/medium/high | ❌ | - |
| gpt-5.4 | ✅ | 1 | jpeg/png/webp/gif | ✅ | ✅ | ✅ | ✅ | low/medium/high | ❌ | - |
| gpt-5-mini | ✅ | 1 | jpeg/png/webp/gif | ✅ | ✅ | ✅ | ✅ | low/medium/high | ❌ | - |
| grok-code-fast-1 | ❌ | - | - | ✅ | ❌ | ✅ | ✅ | - | ❌ | - |
| gpt-5.1 | ✅ | 1 | jpeg/png/webp/gif | ✅ | ✅ | ✅ | ✅ | none/low/medium/high | ❌ | - |
| gpt-5.1-codex-max | ✅ | 1 | jpeg/png/webp/gif | ✅ | ✅ | ✅ | ✅ | low/medium/high | ❌ | - |
| claude-sonnet-4 | ✅ | 5 | jpeg/png/webp | ✅ | ✅ | ✅ | ❌ | - | ❌ | 1,024-32,000 |
| claude-sonnet-4.5 | ✅ | 5 | jpeg/png/webp | ✅ | ✅ | ✅ | ❌ | - | ❌ | 1,024-32,000 |
| claude-opus-4.5 | ✅ | 5 | jpeg/png/webp | ✅ | ✅ | ✅ | ❌ | - | ❌ | 1,024-32,000 |
| claude-haiku-4.5 | ✅ | 5 | jpeg/png/webp | ✅ | ✅ | ✅ | ❌ | - | ❌ | 1,024-32,000 |
| gemini-3-pro-preview | ✅ | 10 | jpeg/png/webp/heic/heif | ✅ | ✅ | ✅ | ❌ | low/high | ❌ | 256-32,000 |
| gemini-3-flash-preview | ✅ | 10 | jpeg/png/webp/heic/heif | ✅ | ✅ | ✅ | ❌ | low/medium/high | ❌ | 256-32,000 |
| gemini-2.5-pro | ✅ | 10 | jpeg/png/webp/heic/heif | ✅ | ✅ | ✅ | ❌ | - | ❌ | 128-32,768 |
| gpt-5.2 | ✅ | 1 | jpeg/png/webp/gif | ✅ | ✅ | ✅ | ✅ | - | ❌ | - |
| gpt-41-copilot | ❌ | - | - | ❌ | ❌ | ✅ | ❌ | - | ❌ | - |
| gpt-4.1 | ✅ | 1 | jpeg/png/webp/gif | ✅ | ✅ | ✅ | ✅ | - | ❌ | - |
| gpt-4o | ✅ | 1 | jpeg/png/webp/gif | ✅ | ✅ | ✅ | ❌ | - | ❌ | - |

> 注：
> - 当前快照中 vision 条目的 `max_prompt_image_size` 在所有支持图片输入的模型上统一为 `3,145,728` 字节（约 3 MiB），因此不再单独列为表列。
> - 本表中的 `❌` 既包括显式 `false`，也包括当前快照未声明该能力；统一按“当前快照未声明支持”处理。

## 四、Picker 模型：访问与端点

| 模型ID | Picker 分类 | 默认聊天 | 回退聊天 | Premium | 计划限制 | 支持端点 |
|---|---|---|---|---|---|---|
| claude-opus-4.6 | powerful | ❌ | ❌ | ✅ | pro/pro_plus/business/enterprise | /v1/messages<br>/chat/completions |
| claude-sonnet-4.6 | versatile | ❌ | ❌ | ✅ | pro/pro_plus/business/enterprise | /chat/completions<br>/v1/messages |
| gemini-3.1-pro-preview | powerful | ❌ | ❌ | ✅ | edu/pro/pro_plus/business/enterprise | /chat/completions |
| gpt-5.2-codex | powerful | ❌ | ❌ | ✅ | pro/edu/pro_plus/business/enterprise | /responses |
| gpt-5.3-codex | powerful | ❌ | ❌ | ✅ | pro/edu/pro_plus/business/enterprise | /responses |
| gpt-5.4 | powerful | ❌ | ❌ | ✅ | pro/pro_plus/business/enterprise | /responses |
| gpt-5-mini | lightweight | ❌ | ❌ | ❌ | - | /chat/completions<br>/responses |
| grok-code-fast-1 | lightweight | ❌ | ❌ | ✅ | - | - |
| gpt-5.1 | versatile | ❌ | ❌ | ✅ | pro/pro_plus/max/business/enterprise/individual_trial/edu | /chat/completions<br>/responses |
| gpt-5.1-codex-max | powerful | ❌ | ❌ | ✅ | pro/pro_plus/max/business/enterprise/individual_trial/edu | /responses |
| claude-sonnet-4 | versatile | ❌ | ❌ | ✅ | pro/pro_plus/max/business/enterprise | /chat/completions<br>/v1/messages |
| claude-sonnet-4.5 | versatile | ❌ | ❌ | ✅ | pro/pro_plus/max/business/enterprise | /chat/completions<br>/v1/messages |
| claude-opus-4.5 | powerful | ❌ | ❌ | ✅ | pro/pro_plus/max/business/enterprise | /chat/completions<br>/v1/messages |
| claude-haiku-4.5 | versatile | ❌ | ❌ | ✅ | - | /chat/completions<br>/v1/messages |
| gemini-3-pro-preview | powerful | ❌ | ❌ | ✅ | pro/pro_plus/max/business/enterprise/individual_trial/edu | - |
| gemini-3-flash-preview | lightweight | ❌ | ❌ | ✅ | pro/pro_plus/max/business/enterprise/individual_trial/edu | - |
| gemini-2.5-pro | powerful | ❌ | ❌ | ✅ | pro/pro_plus/max/business/enterprise/individual_trial/edu | - |
| gpt-5.2 | versatile | ✅ | ❌ | ✅ | pro/pro_plus/max/business/enterprise/individual_trial/edu | /chat/completions<br>/responses |
| gpt-41-copilot | versatile | ❌ | ❌ | ❌ | - | - |
| gpt-4.1 | versatile | ❌ | ✅ | ❌ | - | - |
| gpt-4o | versatile | ❌ | ❌ | ❌ | - | - |

> 注：
> - `individual_trial` 是当前快照里真实存在的计划标识，本文按原始数据保留，不额外推断其商业含义。
> - `grok-code-fast-1` 与 `claude-haiku-4.5` 为 `Premium=✅`，但未给出显式 `restricted_to`；这表示“高等级模型”与“具体计划白名单”是两套独立元数据。
> - `gpt-4.1` 是聊天回退模型，但 `supported_endpoints` 字段在当前快照中缺失，因此本表记为 `-`。

## 五、非 Picker 生成模型（固定版本 / 旧版本 / 内部模型）

| 模型ID | version | 模型名称 | 厂商 | 类型 | vision | Max Output | Premium | 计划限制 | 支持端点 | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt-4o-mini-2024-07-18 | gpt-4o-mini-2024-07-18 | GPT-4o mini | Azure OpenAI | chat | ❌ | 4,096 | ❌ | - | - | 被别名引用：`gpt-4o-mini` |
| gpt-4o-2024-11-20 | gpt-4o-2024-11-20 | GPT-4o | Azure OpenAI | chat | ✅ | 16,384 | ❌ | - | - | 被别名引用：`gpt-4o` |
| gpt-4o-2024-08-06 | gpt-4o-2024-08-06 | GPT-4o | Azure OpenAI | chat | ❌ | 16,384 | ❌ | - | - | - |
| gpt-4.1-2025-04-14 | gpt-4.1-2025-04-14 | GPT-4.1 | Azure OpenAI | chat | ✅ | 16,384 | ❌ | - | - | 被别名引用：`gpt-4.1` |
| gpt-3.5-turbo-0613 | gpt-3.5-turbo-0613 | GPT 3.5 Turbo | Azure OpenAI | chat | ❌ | 4,096 | ❌ | - | - | 被别名引用：`gpt-3.5-turbo` |
| gpt-4 | gpt-4-0613 | GPT 4 | Azure OpenAI | chat | ❌ | 4,096 | ❌ | - | - | 源数据存在重复记录；别名 -> `gpt-4-0613` |
| gpt-4-0613 | gpt-4-0613 | GPT 4 | Azure OpenAI | chat | ❌ | 4,096 | ❌ | - | - | 被别名引用：`gpt-4` |
| gpt-4o-2024-05-13 | gpt-4o-2024-05-13 | GPT-4o | Azure OpenAI | chat | ✅ | 4,096 | ❌ | - | - | 被别名引用：`gpt-4-o-preview` |
| gpt-4-o-preview | gpt-4o-2024-05-13 | GPT-4o | Azure OpenAI | chat | ❌ | 4,096 | ❌ | - | - | 源数据存在重复记录；别名 -> `gpt-4o-2024-05-13`；当前条目未声明 vision，但同 version 固定版本声明了 vision |
| gpt-3.5-turbo | gpt-3.5-turbo-0613 | GPT 3.5 Turbo | Azure OpenAI | chat | ❌ | 4,096 | ❌ | pro/pro_plus/max/business/enterprise/individual_trial/edu | - | 别名 -> `gpt-3.5-turbo-0613`；非 premium，但 billing.restricted_to 仍存在 |
| gpt-4o-mini | gpt-4o-mini-2024-07-18 | GPT-4o mini | Azure OpenAI | chat | ❌ | 4,096 | ❌ | - | - | 别名 -> `gpt-4o-mini-2024-07-18` |

> 注：`gpt-4-o-preview` 的 `version` 指向 `gpt-4o-2024-05-13`，但该条目自身并未声明 vision；本文按原始数据保留，不擅自做能力继承。

## 六、历史条目（承接上一版文档）

> 以下模型来自上一版文档，为保留变更痕迹继续展示；它们未出现在当前 `local/model.json` 快照中，因此按历史条目处理，并使用删除线标记。除非 GitHub 官方另有说明，这里不直接表述为“官方下架”。

| 模型ID | 模型名称 | 厂商 | 版本 | 上次记录上下文窗口 | 上次记录最大输出 | 备注 |
|---|---|---|---|---|---|---|
| ~~gpt-5~~ | ~~GPT-5 (Preview)~~ | ~~Azure OpenAI~~ | ~~gpt-5~~ | ~~128,000~~ | ~~64,000~~ | 上一版文档中存在，但未出现在当前快照中 |
| ~~o3-mini~~ | ~~o3-mini~~ | ~~Azure OpenAI~~ | ~~o3-mini-2025-01-31~~ | ~~200,000~~ | ~~100,000~~ | 上一版文档中存在，但未出现在当前快照中 |
| ~~o3-mini-2025-01-31~~ | ~~o3-mini~~ | ~~Azure OpenAI~~ | ~~o3-mini-2025-01-31~~ | ~~200,000~~ | ~~100,000~~ | 上一版文档中存在，但未出现在当前快照中 |
| ~~o3-mini-paygo~~ | ~~o3-mini~~ | ~~Azure OpenAI~~ | ~~o3-mini-paygo~~ | ~~200,000~~ | ~~100,000~~ | 上一版文档中存在，但未出现在当前快照中 |
| ~~gpt-4o-copilot~~ | ~~GPT-4o Copilot~~ | ~~Azure OpenAI~~ | ~~gpt-4o-copilot~~ | ~~-~~ | ~~-~~ | 上一版文档中存在，但未出现在当前快照中 |
| ~~claude-3.5-sonnet~~ | ~~Claude Sonnet 3.5~~ | ~~Anthropic~~ | ~~claude-3.5-sonnet~~ | ~~90,000~~ | ~~8,192~~ | 上一版文档中存在，但未出现在当前快照中 |
| ~~claude-3.7-sonnet~~ | ~~Claude Sonnet 3.7~~ | ~~Anthropic~~ | ~~claude-3.7-sonnet~~ | ~~200,000~~ | ~~16,384~~ | 上一版文档中存在，但未出现在当前快照中 |
| ~~claude-3.7-sonnet-thought~~ | ~~Claude Sonnet 3.7 Thinking~~ | ~~Anthropic~~ | ~~claude-3.7-sonnet-thought~~ | ~~200,000~~ | ~~16,384~~ | 上一版文档中存在，但未出现在当前快照中 |
| ~~gemini-2.0-flash-001~~ | ~~Gemini 2.0 Flash~~ | ~~Google~~ | ~~gemini-2.0-flash-001~~ | ~~1,000,000~~ | ~~8,192~~ | 上一版文档中存在，但未出现在当前快照中 |
| ~~o4-mini~~ | ~~o4-mini (Preview)~~ | ~~Azure OpenAI~~ | ~~o4-mini-2025-04-16~~ | ~~128,000~~ | ~~16,384~~ | 上一版文档中存在，但未出现在当前快照中 |
| ~~o4-mini-2025-04-16~~ | ~~o4-mini (Preview)~~ | ~~OpenAI~~ | ~~o4-mini-2025-04-16~~ | ~~200,000~~ | ~~100,000~~ | 上一版文档中存在，但未出现在当前快照中 |

## 七、嵌入模型

| 模型ID | 模型名称 | 厂商 | version | 最大输入数 | 支持自定义维度 | Tokenizer | 备注 |
|---|---|---|---|---|---|---|---|
| text-embedding-3-small | Embedding V3 small | Azure OpenAI | text-embedding-3-small | 512 | ✅ | cl100k_base | 按原始数据保留 |
| text-embedding-3-small-inference | Embedding V3 small (Inference) | Azure OpenAI | text-embedding-3-small | - | ✅ | cl100k_base | 按原始数据保留 |
| text-embedding-ada-002 | Embedding V2 Ada | Azure OpenAI | text-embedding-3-small | 512 | ❌ | cl100k_base | 源数据中 version 仍为 text-embedding-3-small，按原样保留 |

## 八、Tokenizer 对照

| Tokenizer | 覆盖 family | 说明 |
|---|---|---|
| cl100k_base | text-embedding-3-small<br>gpt-3.5-turbo<br>gpt-4<br>text-embedding-ada-002 | 4 个 family |
| o200k_base | claude-opus-4.6<br>claude-sonnet-4.6<br>gemini-3.1-pro-preview<br>gpt-5.2-codex<br>gpt-5.3-codex<br>gpt-5.4<br>gpt-5-mini<br>gpt-4o-mini<br>gpt-4o<br>grok-code<br>gpt-5.1<br>gpt-5.1-codex-max<br>claude-sonnet-4<br>claude-sonnet-4.5<br>claude-opus-4.5<br>claude-haiku-4.5<br>gemini-3-pro<br>gemini-3-flash<br>gemini-2.5-pro<br>gpt-4.1<br>gpt-5.2 | 21 个 family |
