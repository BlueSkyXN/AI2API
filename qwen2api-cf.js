// 通义千问 OpenAI 兼容代理 - 完整版
// 包括 /v1/models、/v1/chat/completions（流式和非流）、/v1/images/generations 以及图片上传功能
// 把https://chat.qwen.ai/的Cookie中的token字段值作为APIKEY传入使用openai兼容性标准接口使用即可

export default {
  // 内置模型列表（当获取接口失败时使用）
  defaultModels: [
    "qwen-max-latest",
    "qwen-plus-latest",
    "qwen2.5-vl-72b-instruct",
    "qwen2.5-14b-instruct-1m",
    "qvq-72b-preview",
    "qwq-32b-preview",
    "qwen2.5-coder-32b-instruct",
    "qwen-turbo-latest",
    "qwen2.5-72b-instruct"
  ],

  // 主入口：根据 URL 路径分发请求
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const path = url.pathname;
    const apiPrefix = env.API_PREFIX || '';

    if (path === `${apiPrefix}/v1/models`) {
      return this.handleModels(request);
    } else if (path === `${apiPrefix}/v1/chat/completions`) {
      return this.handleChatCompletions(request);
    } else if (path === `${apiPrefix}/v1/images/generations`) {
      return this.handleImageGenerations(request);
    }

    return new Response("Not Found", { status: 404 });
  },

  // 从请求中提取 Authorization token
  getAuthToken(request) {
    const authHeader = request.headers.get('authorization');
    if (!authHeader) return null;
    return authHeader.replace('Bearer ', '');
  },

  // 处理模型列表接口
  async handleModels(request) {
    const authToken = this.getAuthToken(request);
    let modelsList = [];

    if (authToken) {
      try {
        const response = await fetch('https://chat.qwen.ai/api/models', {
          headers: {
            'Authorization': `Bearer ${authToken}`,
            'User-Agent': 'Mozilla/5.0'
          }
        });
        if (response.ok) {
          const data = await response.json();
          modelsList = data.data.map(item => item.id);
        } else {
          modelsList = [...this.defaultModels];
        }
      } catch (e) {
        console.error('获取模型列表失败:', e);
        modelsList = [...this.defaultModels];
      }
    } else {
      modelsList = [...this.defaultModels];
    }

    // 扩展模型列表，增加变种后缀
    const expandedModels = [];
    for (const model of modelsList) {
      expandedModels.push(model);
      expandedModels.push(model + '-thinking');
      expandedModels.push(model + '-search');
      expandedModels.push(model + '-thinking-search');
      expandedModels.push(model + '-draw');
    }

    return new Response(JSON.stringify({
      object: "list",
      data: expandedModels.map(id => ({
        id,
        object: "model",
        created: Date.now(),
        owned_by: "qwen"
      }))
    }), { headers: { 'Content-Type': 'application/json' } });
  },

  // 处理 /v1/chat/completions 接口
  async handleChatCompletions(request) {
    const authToken = this.getAuthToken(request);
    if (!authToken) {
      return new Response(JSON.stringify({
        error: "请提供正确的 Authorization token"
      }), { status: 401, headers: { 'Content-Type': 'application/json' } });
    }

    let body;
    try {
      body = await request.json();
    } catch (error) {
      return new Response(JSON.stringify({
        error: "无效的请求体，请提供有效的JSON"
      }), { status: 400, headers: { 'Content-Type': 'application/json' } });
    }

    const stream = !!body.stream;
    const messages = body.messages || [];
    const requestId = crypto.randomUUID();

    if (!Array.isArray(messages) || messages.length === 0) {
      return new Response(JSON.stringify({
        error: "请提供有效的 messages 数组"
      }), { status: 400, headers: { 'Content-Type': 'application/json' } });
    }

    let modelName = body.model || "qwen-turbo-latest";
    let chatType = "t2t";

    // 如果模型名包含 -draw，则走图像生成流程
    if (modelName.includes('-draw')) {
      return this.handleDrawRequest(messages, modelName, authToken);
    }

    // 如果是 -thinking 模式，则设置思考配置
    if (modelName.includes('-thinking')) {
      modelName = modelName.replace('-thinking', '');
      if (messages[messages.length - 1]) {
        messages[messages.length - 1].feature_config = { thinking_enabled: true };
      }
    }

    // 如果是 -search 模式，则修改 chat_type
    if (modelName.includes('-search')) {
      modelName = modelName.replace('-search', '');
      chatType = "search";
      if (messages[messages.length - 1]) {
        messages[messages.length - 1].chat_type = "search";
      }
    }

    const requestBody = {
      model: modelName,
      messages,
      stream,
      chat_type: chatType,
      id: requestId
    };

    // 处理图片消息（例如上传图片）：
    const lastMessage = messages[messages.length - 1];
    if (Array.isArray(lastMessage?.content)) {
      const imageItem = lastMessage.content.find(item =>
        item.image_url && item.image_url.url
      );
      if (imageItem) {
        const imageId = await this.uploadImage(imageItem.image_url.url, authToken);
        if (imageId) {
          const index = lastMessage.content.findIndex(item =>
            item.image_url && item.image_url.url
          );
          if (index >= 0) {
            lastMessage.content[index] = {
              type: "image",
              image: imageId
            };
          }
        }
      }
    }

    try {
      const response = await fetch('https://chat.qwen.ai/api/chat/completions', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${authToken}`,
          'Content-Type': 'application/json',
          'User-Agent': 'Mozilla/5.0'
        },
        body: JSON.stringify(requestBody)
      });

      if (!response.ok) {
        const errText = await response.text();
        console.error('Qwen 接口调用失败:', response.status, errText);
        return new Response(JSON.stringify({
          error: `请求通义千问API失败: ${response.status}`,
          details: errText
        }), { status: response.status, headers: { 'Content-Type': 'application/json' } });
      }

      if (stream) {
        return this.handleStreamResponse(response, requestId, modelName);
      } else {
        return this.handleNormalResponse(response, requestId, modelName);
      }
    } catch (e) {
      console.error('请求失败:', e);
      return new Response(JSON.stringify({
        error: "请求通义千问API失败，请检查 token 是否正确"
      }), { status: 500, headers: { 'Content-Type': 'application/json' } });
    }
  },

  // ---------------------- 流式响应处理（重构并去重） ----------------------
  async handleStreamResponse(fetchResponse, requestId, modelName) {
    const { readable, writable } = new TransformStream();
    const writer = writable.getWriter();
    const encoder = new TextEncoder();

    // 辅助函数：将 payload 包装为 SSE 格式后写入，并编码成字节
    const sendSSE = async (payload) => {
      await writer.write(encoder.encode(`data: ${payload}\n\n`));
    };

    // 用于去重：记录上一次完整接收到的 delta 内容
    let previousDelta = "";

    const processStream = async () => {
      try {
        const reader = fetchResponse.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';
        let isFirstChunk = true;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunkStr = decoder.decode(value, { stream: true });
          buffer += chunkStr;

          // SSE 消息通常以 "\n\n" 分隔
          const parts = buffer.split('\n\n');
          buffer = parts.pop() || '';

          for (const part of parts) {
            if (!part.trim()) continue;

            const lines = part.split('\n');
            for (const line of lines) {
              if (!line.startsWith('data: ')) continue;

              const dataStr = line.slice('data: '.length).trim();
              if (dataStr === '[DONE]') {
                await sendSSE('[DONE]');
                console.log('收到 [DONE]，流结束');
                break;
              }

              try {
                const jsonData = JSON.parse(dataStr);
                const delta = jsonData?.choices?.[0]?.delta;
                if (!delta) continue;

                let currentDelta = delta.content || "";
                // 去除重复：如果当前内容以上次完整内容为前缀，则只保留新增部分
                let newContent = currentDelta;
                if (previousDelta && currentDelta.startsWith(previousDelta)) {
                  newContent = currentDelta.substring(previousDelta.length);
                }
                previousDelta = currentDelta;
                if (!newContent) continue;

                const openaiChunk = {
                  id: `chatcmpl-${requestId}`,
                  object: 'chat.completion.chunk',
                  created: Date.now(),
                  model: modelName,
                  choices: [
                    {
                      index: 0,
                      delta: isFirstChunk
                        ? { role: 'assistant', content: newContent }
                        : { content: newContent },
                      finish_reason: null
                    }
                  ]
                };

                if (isFirstChunk) isFirstChunk = false;
                await sendSSE(JSON.stringify(openaiChunk));
              } catch (err) {
                console.error('解析 SSE JSON 失败:', dataStr, err);
              }
            }
          }
        }
        await sendSSE('[DONE]');
      } catch (err) {
        console.error('处理 SSE 流时出错:', err);
        const errorChunk = {
          id: `chatcmpl-${requestId}`,
          object: 'chat.completion.chunk',
          created: Date.now(),
          model: modelName,
          choices: [
            {
              index: 0,
              delta: { content: '【流式处理出错，请重试】' },
              finish_reason: 'error'
            }
          ]
        };
        try {
          await writer.write(encoder.encode(`data: ${JSON.stringify(errorChunk)}\n\n`));
          await writer.write(encoder.encode('data: [DONE]\n\n'));
        } catch (_) {}
      } finally {
        await writer.close();
      }
    };

    processStream();
    return new Response(readable, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Connection': 'keep-alive',
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no'
      }
    });
  },

  // ---------------------- 普通（非流）响应 ----------------------
  async handleNormalResponse(fetchResponse, requestId, modelName) {
    try {
      const data = await fetchResponse.json();
      const content = data?.choices?.[0]?.message?.content || '';
      const finishReason = data?.choices?.[0]?.finish_reason || 'stop';

      return new Response(JSON.stringify({
        id: `chatcmpl-${requestId}`,
        object: 'chat.completion',
        created: Date.now(),
        model: modelName,
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              content
            },
            finish_reason: finishReason
          }
        ],
        usage: data?.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      }), { headers: { 'Content-Type': 'application/json' } });
    } catch (e) {
      console.error('解析普通响应失败:', e);
      return new Response(JSON.stringify({
        error: "解析 Qwen 响应出错"
      }), { status: 500, headers: { 'Content-Type': 'application/json' } });
    }
  },

  // ---------------------- 图像生成请求（handleDrawRequest） ----------------------
  async handleDrawRequest(messages, model, authToken) {
    const prompt = messages[messages.length - 1].content;
    const size = '1024*1024';
    const pureModelName = model.replace('-draw', '').replace('-thinking', '').replace('-search', '');

    try {
      // 创建图像生成任务
      const createResponse = await fetch('https://chat.qwen.ai/api/chat/completions', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${authToken}`,
          'Content-Type': 'application/json',
          'User-Agent': 'Mozilla/5.0'
        },
        body: JSON.stringify({
          stream: false,
          incremental_output: true,
          chat_type: "t2i",
          model: pureModelName,
          messages: [
            {
              role: "user",
              content: prompt,
              chat_type: "t2i",
              extra: {},
              feature_config: { thinking_enabled: false }
            }
          ],
          id: crypto.randomUUID(),
          size: size
        })
      });

      if (!createResponse.ok) {
        const errorText = await createResponse.text();
        return new Response(JSON.stringify({
          error: "图像生成任务创建失败",
          details: errorText
        }), {
          status: 500,
          headers: { 'Content-Type': 'application/json' }
        });
      }

      const createData = await createResponse.json();
      let taskId = null;

      // 查找任务ID
      for (const msg of createData.messages) {
        if (msg.role === 'assistant' && msg.extra?.wanx?.task_id) {
          taskId = msg.extra.wanx.task_id;
          break;
        }
      }

      if (!taskId) {
        return new Response(JSON.stringify({
          error: "无法获取图像生成任务ID"
        }), {
          status: 500,
          headers: { 'Content-Type': 'application/json' }
        });
      }

      // 轮询等待图像生成完成（最多 30 次，每次间隔6秒）
      let imageUrl = null;
      for (let i = 0; i < 30; i++) {
        try {
          const statusResponse = await fetch(`https://chat.qwen.ai/api/v1/tasks/status/${taskId}`, {
            headers: {
              'Authorization': `Bearer ${authToken}`,
              'User-Agent': 'Mozilla/5.0'
            }
          });
          if (statusResponse.ok) {
            const statusData = await statusResponse.json();
            if (statusData.content) {
              imageUrl = statusData.content;
              break;
            }
          }
        } catch (error) {
          // 忽略单次错误
        }
        await new Promise(resolve => setTimeout(resolve, 6000));
      }

      if (!imageUrl) {
        return new Response(JSON.stringify({
          error: "图像生成超时"
        }), {
          status: 500,
          headers: { 'Content-Type': 'application/json' }
        });
      }

      // 返回 OpenAI 标准格式的响应（使用 Markdown 格式嵌入图片）
      return new Response(JSON.stringify({
        id: `chatcmpl-${crypto.randomUUID()}`,
        object: "chat.completion",
        created: Date.now(),
        model: model,
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content: `![${imageUrl}](${imageUrl})`
            },
            finish_reason: "stop"
          }
        ],
        usage: {
          prompt_tokens: 1024,
          completion_tokens: 1024,
          total_tokens: 2048
        }
      }), {
        headers: { 'Content-Type': 'application/json' }
      });
    } catch (error) {
      console.error('图像生成失败:', error);
      return new Response(JSON.stringify({
        error: "图像生成请求失败"
      }), {
        status: 500,
        headers: { 'Content-Type': 'application/json' }
      });
    }
  },

  // ---------------------- 图像生成接口（/v1/images/generations） ----------------------
  async handleImageGenerations(request) {
    const authToken = this.getAuthToken(request);
    if (!authToken) {
      return new Response(JSON.stringify({
        error: "请提供正确的 Authorization token"
      }), {
        status: 401,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    let body;
    try {
      body = await request.json();
    } catch (error) {
      return new Response(JSON.stringify({
        error: "无效的请求体，请提供有效的JSON"
      }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    const { model = "qwen-max-latest-draw", prompt, n = 1, size = '1024*1024' } = body;
    const pureModelName = model.replace('-draw', '').replace('-thinking', '').replace('-search', '');

    try {
      // 创建图像生成任务（非流式，incremental_output: true）
      const createResponse = await fetch('https://chat.qwen.ai/api/chat/completions', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${authToken}`,
          'Content-Type': 'application/json',
          'User-Agent': 'Mozilla/5.0'
        },
        body: JSON.stringify({
          stream: false,
          incremental_output: true,
          chat_type: "t2i",
          model: pureModelName,
          messages: [
            {
              role: "user",
              content: prompt,
              chat_type: "t2i",
              extra: {},
              feature_config: { thinking_enabled: false }
            }
          ],
          id: crypto.randomUUID(),
          size: size
        })
      });

      if (!createResponse.ok) {
        const errorText = await createResponse.text();
        return new Response(JSON.stringify({
          error: "图像生成任务创建失败",
          details: errorText
        }), {
          status: 500,
          headers: { 'Content-Type': 'application/json' }
        });
      }

      const createData = await createResponse.json();
      let taskId = null;
      for (const msg of createData.messages) {
        if (msg.role === 'assistant' && msg.extra?.wanx?.task_id) {
          taskId = msg.extra.wanx.task_id;
          break;
        }
      }
      if (!taskId) {
        return new Response(JSON.stringify({
          error: "无法获取图像生成任务ID"
        }), {
          status: 500,
          headers: { 'Content-Type': 'application/json' }
        });
      }

      let imageUrl = null;
      for (let i = 0; i < 30; i++) {
        try {
          const statusResponse = await fetch(`https://chat.qwen.ai/api/v1/tasks/status/${taskId}`, {
            headers: {
              'Authorization': `Bearer ${authToken}`,
              'User-Agent': 'Mozilla/5.0'
            }
          });
          if (statusResponse.ok) {
            const statusData = await statusResponse.json();
            if (statusData.content) {
              imageUrl = statusData.content;
              break;
            }
          }
        } catch (error) {
          // 忽略错误
        }
        await new Promise(resolve => setTimeout(resolve, 6000));
      }

      if (!imageUrl) {
        return new Response(JSON.stringify({
          error: "图像生成超时"
        }), {
          status: 500,
          headers: { 'Content-Type': 'application/json' }
        });
      }

      // 构造 OpenAI 标准格式的响应数据（返回图片列表）
      const images = Array(n).fill().map(() => ({ url: imageUrl }));
      return new Response(JSON.stringify({
        created: Date.now(),
        data: images
      }), {
        headers: { 'Content-Type': 'application/json' }
      });
    } catch (error) {
      console.error('图像生成失败:', error);
      return new Response(JSON.stringify({
        error: "图像生成请求失败"
      }), {
        status: 500,
        headers: { 'Content-Type': 'application/json' }
      });
    }
  },

  // ---------------------- 图片上传接口 ----------------------
  async uploadImage(base64Data, authToken) {
    try {
      // 从 base64 数据中提取图片数据
      const base64Image = base64Data.split(';base64,').pop();
      const imageData = atob(base64Image);
      const arrayBuffer = new ArrayBuffer(imageData.length);
      const uint8Array = new Uint8Array(arrayBuffer);
      for (let i = 0; i < imageData.length; i++) {
        uint8Array[i] = imageData.charCodeAt(i);
      }
      const formData = new FormData();
      const blob = new Blob([uint8Array], { type: 'image/jpeg' });
      formData.append('file', blob, `image-${Date.now()}.jpg`);

      const response = await fetch('https://chat.qwen.ai/api/v1/files/', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${authToken}`,
          'User-Agent': 'Mozilla/5.0'
        },
        body: formData
      });

      if (response.ok) {
        const data = await response.json();
        return data.id;
      }
      return null;
    } catch (error) {
      console.error('图片上传失败:', error);
      return null;
    }
  }
};
