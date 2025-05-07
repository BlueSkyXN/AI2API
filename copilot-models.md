# AI模型配置信息表

## 模型基本信息

| 模型ID | 模型名称 | 厂商 | 版本 | 上下文窗口大小 | 最大输出Token | 预览版 |
|--------|---------|------|------|---------------|-------------|--------|
| gpt-3.5-turbo | GPT 3.5 Turbo | Azure OpenAI | gpt-3.5-turbo-0613 | 16,384 | 4,096 | ❌ |
| gpt-3.5-turbo-0613 | GPT 3.5 Turbo | Azure OpenAI | gpt-3.5-turbo-0613 | 16,384 | 4,096 | ❌ |
| gpt-4o-mini | GPT-4o mini | Azure OpenAI | gpt-4o-mini-2024-07-18 | 128,000 | 4,096 | ❌ |
| gpt-4o-mini-2024-07-18 | GPT-4o mini | Azure OpenAI | gpt-4o-mini-2024-07-18 | 128,000 | 4,096 | ❌ |
| gpt-4 | GPT 4 | Azure OpenAI | gpt-4-0613 | 32,768 | 4,096 | ❌ |
| gpt-4-0613 | GPT 4 | Azure OpenAI | gpt-4-0613 | 32,768 | 4,096 | ❌ |
| gpt-4o | GPT-4o | Azure OpenAI | gpt-4o-2024-11-20 | 128,000 | 16,384 | ❌ |
| gpt-4o-2024-11-20 | GPT-4o | Azure OpenAI | gpt-4o-2024-11-20 | 128,000 | 16,384 | ❌ |
| gpt-4o-2024-05-13 | GPT-4o | Azure OpenAI | gpt-4o-2024-05-13 | 128,000 | 4,096 | ❌ |
| gpt-4-o-preview | GPT-4o | Azure OpenAI | gpt-4o-2024-05-13 | 128,000 | 4,096 | ❌ |
| gpt-4o-2024-08-06 | GPT-4o | Azure OpenAI | gpt-4o-2024-08-06 | 128,000 | 16,384 | ❌ |
| o1 | o1 (Preview) | Azure OpenAI | o1-2024-12-17 | 200,000 | - | ✅ |
| o1-2024-12-17 | o1 (Preview) | Azure OpenAI | o1-2024-12-17 | 200,000 | - | ✅ |
| o3-mini | o3-mini | Azure OpenAI | o3-mini-2025-01-31 | 200,000 | 100,000 | ❌ |
| o3-mini-2025-01-31 | o3-mini | Azure OpenAI | o3-mini-2025-01-31 | 200,000 | 100,000 | ❌ |
| o3-mini-paygo | o3-mini | Azure OpenAI | o3-mini-paygo | 200,000 | 100,000 | ❌ |
| text-embedding-ada-002 | Embedding V2 Ada | Azure OpenAI | text-embedding-3-small | - | - | ❌ |
| text-embedding-3-small | Embedding V3 small | Azure OpenAI | text-embedding-3-small | - | - | ❌ |
| text-embedding-3-small-inference | Embedding V3 small (Inference) | Azure OpenAI | text-embedding-3-small | - | - | ❌ |
| claude-3.5-sonnet | Claude 3.5 Sonnet | Anthropic | claude-3.5-sonnet | 90,000 | 8,192 | ❌ |
| claude-3.7-sonnet | Claude 3.7 Sonnet | Anthropic | claude-3.7-sonnet | 200,000 | 16,384 | ❌ |
| claude-3.7-sonnet-thought | Claude 3.7 Sonnet Thinking | Anthropic | claude-3.7-sonnet-thought | 200,000 | 16,384 | ❌ |
| gemini-2.0-flash-001 | Gemini 2.0 Flash | Google | gemini-2.0-flash-001 | 1,000,000 | 8,192 | ❌ |
| gemini-2.5-pro | Gemini 2.5 Pro (Preview) | Google | gemini-2.5-pro-preview-03-25 | 128,000 | 64,000 | ✅ |
| gemini-2.5-pro-preview-03-25 | Gemini 2.5 Pro (Preview) | Google | gemini-2.5-pro-preview-03-25 | 128,000 | 64,000 | ✅ |
| o4-mini | o4-mini (Preview) | Azure OpenAI | o4-mini-2025-04-16 | 128,000 | 16,384 | ✅ |
| o4-mini-2025-04-16 | o4-mini (Preview) | OpenAI | o4-mini-2025-04-16 | 128,000 | 16,384 | ✅ |
| gpt-4.1 | GPT-4.1 (Preview) | Azure OpenAI | gpt-4.1-2025-04-14 | 128,000 | 16,384 | ✅ |
| gpt-4.1-2025-04-14 | GPT-4.1 (Preview) | OpenAI | gpt-4.1-2025-04-14 | 128,000 | 16,384 | ✅ |

## 模型特殊能力支持情况

| 模型ID | vision | tool_calls | parallel_tool_calls | streaming | structured_outputs | 
|--------|--------|-----------|---------------------|-----------|-------------------|
| gpt-3.5-turbo | ❌ | ✅ | ❌ | ✅ | ❌ |
| gpt-3.5-turbo-0613 | ❌ | ✅ | ❌ | ✅ | ❌ |
| gpt-4o-mini | ❌ | ✅ | ✅ | ✅ | ❌ |
| gpt-4o-mini-2024-07-18 | ❌ | ✅ | ✅ | ✅ | ❌ |
| gpt-4 | ❌ | ✅ | ❌ | ✅ | ❌ |
| gpt-4-0613 | ❌ | ✅ | ❌ | ✅ | ❌ |
| gpt-4o | ✅ | ✅ | ✅ | ✅ | ❌ |
| gpt-4o-2024-11-20 | ✅ | ✅ | ✅ | ✅ | ❌ |
| gpt-4o-2024-05-13 | ✅ | ✅ | ✅ | ✅ | ❌ |
| gpt-4-o-preview | ❌ | ✅ | ✅ | ✅ | ❌ |
| gpt-4o-2024-08-06 | ❌ | ✅ | ✅ | ✅ | ❌ |
| o1 | ❌ | ✅ | ❌ | ❌ | ✅ |
| o1-2024-12-17 | ❌ | ✅ | ❌ | ❌ | ✅ |
| o3-mini | ❌ | ✅ | ❌ | ✅ | ✅ |
| o3-mini-2025-01-31 | ❌ | ✅ | ❌ | ✅ | ✅ |
| o3-mini-paygo | ❌ | ✅ | ❌ | ✅ | ✅ |
| claude-3.5-sonnet | ✅ | ✅ | ✅ | ✅ | ❌ |
| claude-3.7-sonnet | ✅ | ✅ | ✅ | ✅ | ❌ |
| claude-3.7-sonnet-thought | ✅ | ❌ | ❌ | ✅ | ❌ |
| gemini-2.0-flash-001 | ✅ | ✅ | ✅ | ✅ | ❌ |
| gemini-2.5-pro | ✅ | ✅ | ✅ | ✅ | ❌ |
| gemini-2.5-pro-preview-03-25 | ✅ | ✅ | ✅ | ✅ | ❌ |
| o4-mini | ❌ | ✅ | ✅ | ✅ | ✅ |
| o4-mini-2025-04-16 | ❌ | ✅ | ✅ | ✅ | ✅ |
| gpt-4.1 | ✅ | ✅ | ✅ | ✅ | ✅ |
| gpt-4.1-2025-04-14 | ✅ | ✅ | ✅ | ✅ | ✅ |

## 嵌入模型

| 模型ID | 模型名称 | 厂商 | 版本 | 最大输入 | 支持自定义维度 |
|--------|---------|------|------|---------|--------------|
| text-embedding-ada-002 | Embedding V2 Ada | Azure OpenAI | text-embedding-3-small | 512 | ❌ |
| text-embedding-3-small | Embedding V3 small | Azure OpenAI | text-embedding-3-small | 512 | ✅ |
| text-embedding-3-small-inference | Embedding V3 small (Inference) | Azure OpenAI | text-embedding-3-small | - | ✅ |
