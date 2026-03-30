# GitHub Copilot Models

> **数据来源**：`copilot/` 目录下 4 个 JSON 文件，覆盖 2 种计划（Pro / Business）× 2 种模式（Ask / Agent）共 4 个场景。
>
> | 文件 | 计划 | 模式 | 模型数（含重复 ID） |
> |---|---|---|---|
> | `model-pro-ask.json` | Pro | Ask (Chat) | 37 |
> | `model-pro-agent.json` | Pro | Agent | 9 |
> | `model-business-ask.json` | Business | Ask (Chat) | 37 |
> | `model-business-agent.json` | Business | Agent | 9 |
>
> **关键发现**：
> - Pro 与 Business 之间：**仅 token 限制不同**，billing / 能力 / 端点 / Picker 设置完全一致。
> - Ask 与 Agent 之间：Agent 是 Ask 的严格子集（仅 8+1 个模型），且 Agent 不含 `reasoning_effort`、`adaptive_thinking`、`max_non_streaming_output_tokens`。
> - 因此本文采用**标准化分层结构**：静态元数据只记一次，变量数据（token 限制）按 4 场景展开。

---

## A. 场景可用性矩阵

22 个 Picker 可见模型在 4 个场景下的可用性。✅ = 该场景 JSON 中包含此模型。

| # | 模型 ID | Pro Ask | Pro Agent | Biz Ask | Biz Agent |
|---|---|---|---|---|---|
| 1 | claude-opus-4.6 | ✅ | ✅ | ✅ | ✅ |
| 2 | claude-opus-4.5 | ✅ | ✅ | ✅ | ✅ |
| 3 | claude-sonnet-4.6 | ✅ | ✅ | ✅ | ✅ |
| 4 | claude-sonnet-4.5 | ✅ | ✅ | ✅ | ✅ |
| 5 | claude-sonnet-4 | ✅ | - | ✅ | - |
| 6 | claude-haiku-4.5 | ✅ | - | ✅ | - |
| 7 | gpt-5.4 | ✅ | ✅ | ✅ | ✅ |
| 8 | gpt-5.3-codex | ✅ | ✅ | ✅ | ✅ |
| 9 | gpt-5.2-codex | ✅ | ✅ | ✅ | ✅ |
| 10 | gpt-5.1-codex-max | ✅ | ✅ | ✅ | ✅ |
| 11 | gpt-5.2 | ✅ | - | ✅ | - |
| 12 | gpt-5.1 | ✅ | - | ✅ | - |
| 13 | gpt-5.4-mini | ✅ | - | ✅ | - |
| 14 | gpt-5-mini | ✅ | - | ✅ | - |
| 15 | gpt-4.1 | ✅ | - | ✅ | - |
| 16 | gpt-4o | ✅ | - | ✅ | - |
| 17 | gpt-41-copilot | ✅ | - | ✅ | - |
| 18 | gemini-3.1-pro-preview | ✅ | - | ✅ | - |
| 19 | gemini-2.5-pro | ✅ | - | ✅ | - |
| 20 | gemini-3-flash-preview | ✅ | - | ✅ | - |
| 21 | grok-code-fast-1 | ✅ | - | ✅ | - |
| 22 | gpt-5.2 | ✅ | - | ✅ | - |
| - | auto | - | ✅ | - | ✅ |

> **Agent 模式仅含 8 个真实模型**（#1-4, #7-10）+ `auto` 虚拟入口。Pro 与 Business 的 Agent 模型集完全相同。

---

## B. 模型身份信息（静态，跨场景一致）

| 模型 ID | 显示名称 | 厂商 | version | family | 类型 | Tokenizer | Picker 分类 | Preview |
|---|---|---|---|---|---|---|---|---|
| claude-opus-4.6 | Claude Opus 4.6 | Anthropic | claude-opus-4.6 | claude-opus-4.6 | chat | o200k_base | powerful | ❌ |
| claude-opus-4.5 | Claude Opus 4.5 | Anthropic | claude-opus-4.5 | claude-opus-4.5 | chat | o200k_base | powerful | ❌ |
| claude-sonnet-4.6 | Claude Sonnet 4.6 | Anthropic | claude-sonnet-4.6 | claude-sonnet-4.6 | chat | o200k_base | versatile | ❌ |
| claude-sonnet-4.5 | Claude Sonnet 4.5 | Anthropic | claude-sonnet-4.5 | claude-sonnet-4.5 | chat | o200k_base | versatile | ❌ |
| claude-sonnet-4 | Claude Sonnet 4 | Anthropic | claude-sonnet-4 | claude-sonnet-4 | chat | o200k_base | versatile | ❌ |
| claude-haiku-4.5 | Claude Haiku 4.5 | Anthropic | claude-haiku-4.5 | claude-haiku-4.5 | chat | o200k_base | versatile | ❌ |
| gpt-5.4 | GPT-5.4 | OpenAI | gpt-5.4 | gpt-5.4 | chat | o200k_base | powerful | ❌ |
| gpt-5.3-codex | GPT-5.3-Codex | OpenAI | gpt-5.3-codex | gpt-5.3-codex | chat | o200k_base | powerful | ❌ |
| gpt-5.2-codex | GPT-5.2-Codex | OpenAI | gpt-5.2-codex | gpt-5.2-codex | chat | o200k_base | powerful | ❌ |
| gpt-5.1-codex-max | GPT-5.1-Codex-Max | OpenAI | gpt-5.1-codex-max | gpt-5.1-codex-max | chat | o200k_base | powerful | ❌ |
| gpt-5.2 | GPT-5.2 | OpenAI | gpt-5.2 | gpt-5.2 | chat | o200k_base | versatile | ❌ |
| gpt-5.1 | GPT-5.1 | OpenAI | gpt-5.1 | gpt-5.1 | chat | o200k_base | versatile | ❌ |
| gpt-5.4-mini | GPT-5.4 mini | OpenAI | gpt-5.4-mini | gpt-5.4-mini | chat | o200k_base | lightweight | ❌ |
| gpt-5-mini | GPT-5 mini | Azure OpenAI | gpt-5-mini | gpt-5-mini | chat | o200k_base | lightweight | ❌ |
| gpt-4.1 | GPT-4.1 | Azure OpenAI | gpt-4.1-2025-04-14 | gpt-4.1 | chat | o200k_base | versatile | ❌ |
| gpt-4o | GPT-4o | Azure OpenAI | gpt-4o-2024-11-20 | gpt-4o | chat | o200k_base | versatile | ❌ |
| gpt-41-copilot | GPT-4.1 Copilot | Azure OpenAI | gpt-41-copilot | gpt-4.1 | completion | o200k_base | versatile | ❌ |
| gemini-3.1-pro-preview | Gemini 3.1 Pro | Google | gemini-3.1-pro-preview | gemini-3.1-pro-preview | chat | o200k_base | powerful | ✅ |
| gemini-2.5-pro | Gemini 2.5 Pro | Google | gemini-2.5-pro | gemini-2.5-pro | chat | o200k_base | powerful | ❌ |
| gemini-3-flash-preview | Gemini 3 Flash | Google | gemini-3-flash-preview | gemini-3-flash | chat | o200k_base | lightweight | ✅ |
| grok-code-fast-1 | Grok Code Fast 1 | xAI | grok-code-fast-1 | grok-code | chat | o200k_base | lightweight | ❌ |

> **特殊角色**：`gpt-5.2` = 默认聊天模型（`is_chat_default`）；`gpt-4.1` = 回退模型（`is_chat_fallback`）。
> `gpt-41-copilot` 类型为 `completion`，无 token 限制声明。

---

## C. 计费与访问控制（跨场景一致）

| 模型 ID | Premium | 倍率 | 计划白名单（`restricted_to`） | API 端点 |
|---|---|---|---|---|
| claude-opus-4.6 | ✅ | 3 | pro, pro_plus, individual_trial, business, enterprise | /v1/messages · /chat/completions |
| claude-opus-4.5 | ✅ | 3 | pro, pro_plus, max, business, enterprise | /chat/completions · /v1/messages |
| claude-sonnet-4.6 | ✅ | 1 | pro, pro_plus, individual_trial, business, enterprise | /chat/completions · /v1/messages |
| claude-sonnet-4.5 | ✅ | 1 | pro, pro_plus, max, business, enterprise | /chat/completions · /v1/messages |
| claude-sonnet-4 | ✅ | 1 | pro, pro_plus, max, business, enterprise | /chat/completions · /v1/messages |
| claude-haiku-4.5 | ✅ | 0.33 | *(未声明)* | /chat/completions · /v1/messages |
| gpt-5.4 | ✅ | 1 | pro, pro_plus, individual_trial, business, enterprise | /responses |
| gpt-5.3-codex | ✅ | 1 | pro, edu, pro_plus, individual_trial, business, enterprise | /responses |
| gpt-5.2-codex | ✅ | 1 | pro, edu, pro_plus, individual_trial, business, enterprise | /responses |
| gpt-5.1-codex-max | ✅ | 1 | pro, pro_plus, max, business, enterprise, individual_trial, edu | /responses |
| gpt-5.2 | ✅ | 1 | pro, pro_plus, max, business, enterprise, individual_trial, edu | /chat/completions · /responses |
| gpt-5.1 | ✅ | 1 | pro, pro_plus, max, business, enterprise, individual_trial, edu | /chat/completions · /responses |
| gpt-5.4-mini | ✅ | 0.33 | pro, pro_plus, individual_trial, edu, business, enterprise | /responses |
| gpt-5-mini | ❌ | 0 | *(未限制)* | /chat/completions · /responses |
| gpt-4.1 | ❌ | 0 | *(未限制)* | - |
| gpt-4o | ❌ | 0 | *(未限制)* | - |
| gpt-41-copilot | ❌ | 0 | *(未限制)* | - |
| gemini-3.1-pro-preview | ✅ | 1 | edu, pro, pro_plus, individual_trial, business, enterprise | /chat/completions |
| gemini-2.5-pro | ✅ | 1 | pro, pro_plus, max, business, enterprise, individual_trial, edu | - |
| gemini-3-flash-preview | ✅ | 0.33 | pro, pro_plus, max, business, enterprise, individual_trial, edu | - |
| grok-code-fast-1 | ✅ | 0.25 | *(未声明)* | - |

---

## D. Token 限制（按 4 场景展开）

这是 Pro 与 Business 之间**唯一的差异维度**。Agent 场景额外不含 `max_non_streaming_output_tokens`。

### D-1. 限制完全一致的模型（4 场景相同）

以下模型在所有可用场景中 token 限制完全相同，无需区分计划和模式。

| 模型 ID | 上下文窗口 | Max Prompt | Max Output |
|---|---|---|---|
| gpt-5.4 | 400,000 | 272,000 | 128,000 |
| gpt-5.3-codex | 400,000 | 272,000 | 128,000 |
| gpt-5.2-codex | 400,000 | 272,000 | 128,000 |
| gpt-5.4-mini | 400,000 | 272,000 | 128,000 |
| gpt-5.1-codex-max | 400,000 | 128,000 | 128,000 |
| gpt-5.1 | 264,000 | 128,000 | 64,000 |
| gpt-5-mini | 264,000 | 128,000 | 64,000 |
| claude-sonnet-4 | 216,000 | 128,000 | 16,000 |
| gemini-2.5-pro | 128,000 | 128,000 | 64,000 |
| gemini-3-flash-preview | 128,000 | 128,000 | 64,000 |
| gpt-4.1 | 128,000 | 128,000 | 16,384 |
| gpt-4o | 128,000 | 64,000 | 4,096 |

### D-2. Pro / Business 限制不同的模型

以下模型在 Pro 和 Business 之间存在 token 限制差异。Ask 与 Agent 的限制在同一计划内相同（Agent 仅缺少非流式输出列）。

| 模型 ID | 字段 | Pro | Business |
|---|---|---|---|
| **claude-opus-4.6** | 上下文窗口 | 144,000 | 200,000 |
| | Max Prompt | 128,000 | 168,000 |
| | Max Output | 64,000 | 32,000 |
| | 非流式 Output | 16,000 | 16,000 |
| **claude-sonnet-4.6** | 上下文窗口 | 200,000 | 200,000 |
| | Max Prompt | 128,000 | 168,000 |
| | Max Output | 32,000 | 32,000 |
| | 非流式 Output | 16,000 | 16,000 |
| **claude-sonnet-4.5** | 上下文窗口 | 144,000 | 200,000 |
| | Max Prompt | 128,000 | 168,000 |
| | Max Output | 32,000 | 32,000 |
| | 非流式 Output | 16,000 | 16,000 |
| **claude-opus-4.5** | 上下文窗口 | 160,000 | 200,000 |
| | Max Prompt | 128,000 | 168,000 |
| | Max Output | 32,000 | 32,000 |
| | 非流式 Output | 16,000 | 16,000 |
| **claude-haiku-4.5** | 上下文窗口 | 144,000 | 200,000 |
| | Max Prompt | 128,000 | 136,000 |
| | Max Output | 32,000 | 64,000 |
| | 非流式 Output | 16,000 | 16,000 |
| **gemini-3.1-pro-preview** | 上下文窗口 | 128,000 | 200,000 |
| | Max Prompt | 128,000 | 136,000 |
| | Max Output | 64,000 | 64,000 |
| **gpt-5.2** | 上下文窗口 | 264,000 | 400,000 |
| | Max Prompt | 128,000 | 272,000 |
| | Max Output | 64,000 | 128,000 |
| **grok-code-fast-1** | 上下文窗口 | 128,000 | 256,000 |
| | Max Prompt | 128,000 | 192,000 |
| | Max Output | 64,000 | 64,000 |

> **规律**：Business 计划普遍获得更大的上下文窗口和 Prompt 空间。`claude-opus-4.6` 和 `claude-haiku-4.5` 是例外——它们的 Max Output 在两个计划间相反方向变化。

---

## E. 模型能力（跨场景一致，Agent 标注除外）

### E-1. 视觉能力

所有支持视觉的模型，`max_prompt_image_size` 统一为 **3,145,728 字节**（≈ 3 MiB）。

| 模型 ID | Vision | 最大图片数 | 支持格式 |
|---|---|---|---|
| claude-opus-4.6 | ✅ | 1 | jpeg, png, webp |
| claude-opus-4.5 | ✅ | 5 | jpeg, png, webp |
| claude-sonnet-4.6 | ✅ | 5 | jpeg, png, webp |
| claude-sonnet-4.5 | ✅ | 5 | jpeg, png, webp |
| claude-sonnet-4 | ✅ | 5 | jpeg, png, webp |
| claude-haiku-4.5 | ✅ | 5 | jpeg, png, webp |
| gpt-5.4 | ✅ | 1 | jpeg, png, webp, gif |
| gpt-5.3-codex | ✅ | 1 | jpeg, png, webp, gif |
| gpt-5.2-codex | ✅ | 1 | jpeg, png, webp, gif |
| gpt-5.1-codex-max | ✅ | 1 | jpeg, png, webp, gif |
| gpt-5.2 | ✅ | 1 | jpeg, png, webp, gif |
| gpt-5.1 | ✅ | 1 | jpeg, png, webp, gif |
| gpt-5.4-mini | ✅ | 1 | jpeg, png, webp, gif |
| gpt-5-mini | ✅ | 1 | jpeg, png, webp, gif |
| gpt-4.1 | ✅ | 1 | jpeg, png, webp, gif |
| gpt-4o | ✅ | 1 | jpeg, png, webp, gif |
| gemini-3.1-pro-preview | ✅ | 10 | jpeg, png, webp |
| gemini-2.5-pro | ✅ | 10 | jpeg, png, webp, heic, heif |
| gemini-3-flash-preview | ✅ | 10 | jpeg, png, webp, heic, heif |
| gpt-41-copilot | ❌ | - | - |
| grok-code-fast-1 | ❌ | - | - |

### E-2. 工具与输出能力

| 模型 ID | Tool Calls | 并行工具调用 | Streaming | 结构化输出 |
|---|---|---|---|---|
| claude-opus-4.6 | ✅ | ✅ | ✅ | ✅ |
| claude-opus-4.5 | ✅ | ✅ | ✅ | ❌ |
| claude-sonnet-4.6 | ✅ | ✅ | ✅ | ✅ |
| claude-sonnet-4.5 | ✅ | ✅ | ✅ | ❌ |
| claude-sonnet-4 | ✅ | ✅ | ✅ | ❌ |
| claude-haiku-4.5 | ✅ | ✅ | ✅ | ❌ |
| gpt-5.4 | ✅ | ✅ | ✅ | ✅ |
| gpt-5.3-codex | ✅ | ✅ | ✅ | ✅ |
| gpt-5.2-codex | ✅ | ✅ | ✅ | ✅ |
| gpt-5.1-codex-max | ✅ | ✅ | ✅ | ✅ |
| gpt-5.2 | ✅ | ✅ | ✅ | ✅ |
| gpt-5.1 | ✅ | ✅ | ✅ | ✅ |
| gpt-5.4-mini | ✅ | ✅ | ✅ | ✅ |
| gpt-5-mini | ✅ | ✅ | ✅ | ✅ |
| gpt-4.1 | ✅ | ✅ | ✅ | ✅ |
| gpt-4o | ✅ | ✅ | ✅ | ❌ |
| gpt-41-copilot | ❌ | ❌ | ✅ | ❌ |
| gemini-3.1-pro-preview | ✅ | ✅ | ✅ | ❌ |
| gemini-2.5-pro | ✅ | ✅ | ✅ | ❌ |
| gemini-3-flash-preview | ✅ | ✅ | ✅ | ❌ |
| grok-code-fast-1 | ✅ | ❌ | ✅ | ✅ |

### E-3. 推理与思考能力

`reasoning_effort` 和 `adaptive_thinking` **仅存在于 Ask 场景**，Agent 场景不含这两个字段。

| 模型 ID | Reasoning Effort（Ask） | Adaptive Thinking（Ask） | Thinking Budget |
|---|---|---|---|
| claude-opus-4.6 | low / medium / high | ✅ | 1,024 – 32,000 |
| claude-sonnet-4.6 | low / medium / high | ✅ | 1,024 – 32,000 |
| claude-opus-4.5 | - | - | 1,024 – 32,000 |
| claude-sonnet-4.5 | - | - | 1,024 – 32,000 |
| claude-sonnet-4 | - | - | 1,024 – 32,000 |
| claude-haiku-4.5 | - | - | 1,024 – 32,000 |
| gpt-5.4 | low / medium / high / xhigh | - | - |
| gpt-5.3-codex | low / medium / high / xhigh | - | - |
| gpt-5.2-codex | low / medium / high / xhigh | - | - |
| gpt-5.1-codex-max | low / medium / high / xhigh | - | - |
| gpt-5.2 | low / medium / high / xhigh | - | - |
| gpt-5.1 | none / low / medium / high | - | - |
| gpt-5.4-mini | low / medium / high | - | - |
| gpt-5-mini | low / medium / high | - | - |
| gemini-3.1-pro-preview | low / medium / high | - | 256 – 32,000 |
| gemini-3-flash-preview | low / medium / high | - | 256 – 32,000 |
| gemini-2.5-pro | - | - | 128 – 32,768 |

> 未列出的模型（gpt-4.1, gpt-4o, gpt-41-copilot, grok-code-fast-1）不支持 reasoning_effort 和 thinking budget。

---

## F. 非 Picker 模型（旧版本 / 固定版本 / 内部）

以下模型 `model_picker_enabled: false`，仅出现在 Ask 场景中，Pro 与 Business 完全一致。

| 模型 ID | 名称 | 厂商 | version | 上下文窗口 | Max Output | Vision | 说明 |
|---|---|---|---|---|---|---|---|
| gpt-4o-2024-11-20 | GPT-4o | Azure OpenAI | gpt-4o-2024-11-20 | 128,000 | 16,384 | ✅ | 被 `gpt-4o` 引用 |
| gpt-4o-2024-08-06 | GPT-4o | Azure OpenAI | gpt-4o-2024-08-06 | 128,000 | 16,384 | ❌ | - |
| gpt-4o-2024-05-13 | GPT-4o | Azure OpenAI | gpt-4o-2024-05-13 | 128,000 | 4,096 | ✅ | 被 `gpt-4-o-preview` 引用 |
| gpt-4o-mini-2024-07-18 | GPT-4o mini | Azure OpenAI | gpt-4o-mini-2024-07-18 | 128,000 | 4,096 | ❌ | 被 `gpt-4o-mini` 引用 |
| gpt-4o-mini | GPT-4o mini | Azure OpenAI | gpt-4o-mini-2024-07-18 | 128,000 | 4,096 | ❌ | 别名 → gpt-4o-mini-2024-07-18 |
| gpt-4.1-2025-04-14 | GPT-4.1 | Azure OpenAI | gpt-4.1-2025-04-14 | 128,000 | 16,384 | ✅ | 被 `gpt-4.1` 引用 |
| gpt-4-0613 | GPT 4 | Azure OpenAI | gpt-4-0613 | 32,768 | 4,096 | ❌ | 被 `gpt-4` 引用 |
| gpt-4 | GPT 4 | Azure OpenAI | gpt-4-0613 | 32,768 | 4,096 | ❌ | 别名 → gpt-4-0613；源数据有重复 |
| gpt-4-o-preview | GPT-4o | Azure OpenAI | gpt-4o-2024-05-13 | 128,000 | 4,096 | ❌ | 别名 → gpt-4o-2024-05-13；自身未声明 vision |
| gpt-3.5-turbo-0613 | GPT 3.5 Turbo | Azure OpenAI | gpt-3.5-turbo-0613 | 16,384 | 4,096 | ❌ | 被 `gpt-3.5-turbo` 引用 |
| gpt-3.5-turbo | GPT 3.5 Turbo | Azure OpenAI | gpt-3.5-turbo-0613 | 16,384 | 4,096 | ❌ | 别名 → gpt-3.5-turbo-0613 |

### 特殊条目

| 模型 ID | 名称 | 厂商 | 场景 | 说明 |
|---|---|---|---|---|
| auto | Auto | GitHub | 仅 Agent | 虚拟入口，自动选择底层模型，multiplier=1 |
| goldeneye-free-auto | Goldeneye | Azure OpenAI | 仅 Pro Ask | family=oswe-vscode-large, Preview, ctx=400k, output=128k, restricted_to 含 free |

---

## G. 嵌入模型（仅 Ask 场景，Pro / Business 一致）

| 模型 ID | 名称 | 厂商 | version | 最大输入数 | 自定义维度 | Tokenizer |
|---|---|---|---|---|---|---|
| text-embedding-3-small | Embedding V3 small | Azure OpenAI | text-embedding-3-small | 512 | ✅ | cl100k_base |
| text-embedding-3-small-inference | Embedding V3 small (Inference) | Azure OpenAI | text-embedding-3-small | - | ✅ | cl100k_base |
| text-embedding-ada-002 | Embedding V2 Ada | Azure OpenAI | text-embedding-3-small | 512 | ❌ | cl100k_base |

---

## H. 历史条目

> 以下模型存在于旧版文档中，当前 4 份 JSON 快照均未出现。

| 模型 ID | 名称 | 厂商 |
|---|---|---|
| ~~gpt-5~~ | ~~GPT-5 (Preview)~~ | ~~Azure OpenAI~~ |
| ~~o3-mini~~ | ~~o3-mini~~ | ~~Azure OpenAI~~ |
| ~~o3-mini-2025-01-31~~ | ~~o3-mini~~ | ~~Azure OpenAI~~ |
| ~~o3-mini-paygo~~ | ~~o3-mini~~ | ~~Azure OpenAI~~ |
| ~~gpt-4o-copilot~~ | ~~GPT-4o Copilot~~ | ~~Azure OpenAI~~ |
| ~~claude-3.5-sonnet~~ | ~~Claude Sonnet 3.5~~ | ~~Anthropic~~ |
| ~~claude-3.7-sonnet~~ | ~~Claude Sonnet 3.7~~ | ~~Anthropic~~ |
| ~~claude-3.7-sonnet-thought~~ | ~~Claude Sonnet 3.7 Thinking~~ | ~~Anthropic~~ |
| ~~gemini-2.0-flash-001~~ | ~~Gemini 2.0 Flash~~ | ~~Google~~ |
| ~~gemini-3-pro-preview~~ | ~~Gemini 3 Pro (Preview)~~ | ~~Google~~ |
| ~~o4-mini~~ | ~~o4-mini (Preview)~~ | ~~Azure OpenAI~~ |
| ~~o4-mini-2025-04-16~~ | ~~o4-mini (Preview)~~ | ~~OpenAI~~ |
