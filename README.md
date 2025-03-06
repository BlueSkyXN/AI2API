# Qwen2API
## 项目简介

Qwen2API，用于将通义千问(Qwen AI)的WEB转换为OpenAI兼容的API接口格式，让您可以通过标准的OpenAI API调用方式来使用通义千问模型。该代理支持包括模型列表查询、聊天补全（流式和非流式）、图像生成，以及图片上传功能，为开发者提供了便捷的集成方式。

## 特性

- **OpenAI API兼容**: 提供与OpenAI API格式兼容的接口，方便现有OpenAI项目迁移
- **模型支持**: 支持通义千问的各类模型，包括qwen-max、qwen-plus等
- **模型变体**: 自动扩展模型名称，支持以下后缀功能:
  - `-thinking`: 启用思考模式
  - `-search`: 启用搜索增强
  - `-draw`: 启用图像生成 【可能存在问题】
  - 以上后缀可组合使用，如`qwen-max-latest-thinking-search`
- **流式输出**: 支持流式响应，减少首字等待时间
- **多模态交互**: 支持图片上传和图像生成【可能存在问题】
- **图像生成**: 提供专用的图像生成接口【可能存在问题】

## 部署要求

- CloudFlare账号
- CloudFlare Workers服务

## 安装部署

1. 登录CloudFlare Workers控制台
2. 创建新的Worker
3. 将[qwen2api-cf.js](qwen2api-cf.js)代码复制到Worker编辑器中
4. 保存并部署

## 配置选项

您可以通过环境变量配置以下选项:

| 变量名 | 描述 | 默认值 |
|-------|------|--------|
| API_PREFIX | API路径前缀，可用于自定义路由 | 空字符串 |

## 使用方法

### 认证

使用通义千问的Token作为API密钥，在请求头中设置`Authorization: Bearer {YOUR_QWEN_TOKEN}`。

**获取Token方法**:
1. 访问[通义千问官网](https://chat.qwen.ai/)
2. 登录您的账号
3. 从Cookie中提取`token`字段的值

### 支持的API端点

#### 1. 获取模型列表

```
GET /v1/models
```

**响应示例**:
```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen-max-latest",
      "object": "model",
      "created": 1709128113453,
      "owned_by": "qwen"
    },
    {
      "id": "qwen-max-latest-thinking",
      "object": "model",
      "created": 1709128113453,
      "owned_by": "qwen"
    },
    // 更多模型...
  ]
}
```

#### 2. 聊天补全

```
POST /v1/chat/completions
```

**请求体示例**:
```json
{
  "model": "qwen-max-latest",
  "messages": [
    {
      "role": "user",
      "content": "你好，请介绍一下自己"
    }
  ],
  "stream": false
}
```

**特殊功能**:
- 使用`-thinking`后缀启用思考模式
- 使用`-search`后缀启用搜索增强
- 同时传递图片(多模态)

**多模态示例**:
```json
{
  "model": "qwen2.5-vl-72b-instruct",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "这张图片是什么?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,/9j/4AAQ..."
          }
        }
      ]
    }
  ]
}
```

#### 3. 图像生成

**方法1**: 使用聊天接口

```
POST /v1/chat/completions
```

```json
{
  "model": "qwen-max-latest-draw",
  "messages": [
    {
      "role": "user",
      "content": "画一只可爱的猫咪"
    }
  ]
}
```

**方法2**: 使用专用图像生成接口

```
POST /v1/images/generations
```

```json
{
  "model": "qwen-max-latest-draw",
  "prompt": "画一只可爱的猫咪",
  "n": 1,
  "size": "1024*1024"
}
```

## 支持的模型

系统内置了以下默认模型，当API获取失败时会使用这些模型：

- qwen-max-latest
- qwen-plus-latest
- qwen2.5-vl-72b-instruct
- qwen2.5-14b-instruct-1m
- qvq-72b-preview
- qwq-32b-preview
- qwen2.5-coder-32b-instruct
- qwen-turbo-latest
- qwen2.5-72b-instruct

每个模型都支持添加后缀变体(-thinking、-search、-draw)。

## 技术实现细节

### 架构概述

该代理作为中间层，将OpenAI格式的请求转换为通义千问API格式，并将通义千问的响应转换回OpenAI格式。主要处理流程包括：

1. 解析请求和提取Token
2. 根据URL路径分发到不同处理函数
3. 转换请求格式并调用通义千问API
4. 处理响应并转换格式
5. 特殊处理流式响应和图像生成

### 模型处理机制

- 基础模型名称处理：从请求中提取模型名称
- 后缀功能处理：解析后缀并应用相应的配置
  - `-thinking`: 设置`feature_config.thinking_enabled = true`
  - `-search`: 设置`chat_type = "search"`
  - `-draw`: 切换到图像生成流程

### 流式响应处理

代理实现了高效的流式响应处理机制：
1. 使用TransformStream处理数据流
2. 对数据进行SSE(Server-Sent Events)格式转换
3. 实现增量去重逻辑，确保内容不重复
4. 处理完成标记和结束流

### 图像生成实现

图像生成使用任务创建和状态轮询机制：
1. 创建图像生成任务并获取taskId
2. 定期轮询任务状态(最多30次，每6秒一次)
3. 获取生成的图像URL并返回

## 常见问题与解决方案

### Token无效或过期

**症状**: 请求返回401错误
**解决方案**: 重新获取通义千问Cookie中的token值

### 模型列表获取失败

**症状**: 仅显示默认模型列表
**解决方案**: 检查网络连接和token有效性

### 图像生成超时

**症状**: 返回"图像生成超时"错误
**解决方案**: 
- 检查网络连接
- 尝试简化图像描述
- 尝试减小图像尺寸

### 流式响应中断

**症状**: 响应突然停止
**解决方案**: 
- 检查网络稳定性
- 减少请求的复杂度

