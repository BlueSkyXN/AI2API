#!/usr/bin/env python3
"""
OpenAI to Augment API Adapter

这个FastAPI应用程序将OpenAI API请求格式转换为Augment API格式，
允许OpenAI客户端直接与Augment服务通信。
所有配置参数都通过命令行参数提供，不依赖于环境变量或配置文件。
"""

import os
import json
import uuid
import time
import logging
import argparse
from typing import List, Optional, Dict, Any, Literal, Union
from datetime import datetime

import httpx
from fastapi import FastAPI, Header, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

#################################################
# 模型定义
#################################################

# 新增：支持OpenAI新格式的内容项定义
class ContentItem(BaseModel):
    """表示OpenAI聊天API中的内容项"""
    type: str  # 例如 "text", "image_url" 等
    text: Optional[str] = None
    # 可以在这里添加其他内容类型的字段，如image_url等

# OpenAI API 请求模型
class ChatMessage(BaseModel):
    """表示OpenAI聊天API中的单条消息"""
    role: Literal["system", "user", "assistant", "function"]
    # 修改：content字段现在可以是字符串或内容项数组
    content: Optional[Union[str, List[ContentItem]]] = None
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    """OpenAI聊天完成API请求模型"""
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    user: Optional[str] = None

# OpenAI API 响应模型
class ChatCompletionResponseChoice(BaseModel):
    """OpenAI聊天完成API响应中的单个选择"""
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None

class Usage(BaseModel):
    """OpenAI API响应中的token使用信息"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    """OpenAI聊天完成API响应模型"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Usage

# OpenAI API 流式响应模型
class ChatCompletionStreamResponseChoice(BaseModel):
    """OpenAI聊天完成流式API响应中的单个选择"""
    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None

class ChatCompletionStreamResponse(BaseModel):
    """OpenAI聊天完成流式API响应模型"""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamResponseChoice]

# 模型信息响应
class ModelInfo(BaseModel):
    """OpenAI模型信息"""
    id: str
    object: str = "model"
    created: int
    owned_by: str = "augment"

class ModelListResponse(BaseModel):
    """OpenAI模型列表响应"""
    object: str = "list"
    data: List[ModelInfo]

# Augment API 请求相关模型
class AugmentResponseNode(BaseModel):
    """Augment API响应节点"""
    id: int
    type: int
    content: str
    tool_use: Optional[Any] = None

class AugmentChatHistoryItem(BaseModel):
    """Augment API聊天历史记录条目"""
    request_message: str
    response_text: str
    request_id: Optional[str] = None
    request_nodes: List[Any] = []
    response_nodes: List[AugmentResponseNode] = []

class AugmentBlobs(BaseModel):
    """Augment API Blobs对象"""
    checkpoint_id: Optional[str] = None
    added_blobs: List[Any] = []
    deleted_blobs: List[Any] = []

class AugmentVcsChange(BaseModel):
    """Augment API VCS更改"""
    working_directory_changes: List[Any] = []

class AugmentFeatureFlags(BaseModel):
    """Augment API功能标志"""
    support_raw_output: bool = True

# 完整的Augment API请求模型
class AugmentChatRequest(BaseModel):
    """Augment API聊天请求模型 - 基于抓包分析更新"""
    model: Optional[str] = None
    path: Optional[str] = None
    prefix: Optional[str] = None
    selected_code: Optional[str] = None
    suffix: Optional[str] = None
    message: str
    chat_history: List[AugmentChatHistoryItem] = []
    lang: Optional[str] = None
    blobs: AugmentBlobs = AugmentBlobs()
    user_guided_blobs: List[Any] = []
    context_code_exchange_request_id: Optional[str] = None
    vcs_change: AugmentVcsChange = AugmentVcsChange()
    recency_info_recent_changes: List[Any] = []
    external_source_ids: List[Any] = []
    disable_auto_external_sources: Optional[bool] = None
    user_guidelines: str = ""
    workspace_guidelines: str = ""
    feature_detection_flags: AugmentFeatureFlags = AugmentFeatureFlags()
    tool_definitions: List[Any] = []
    nodes: List[Any] = []
    mode: str = "CHAT"
    agent_memories: Optional[Any] = None
    system_prompt: Optional[str] = None  # 保留此字段以兼容之前的代码

# Augment API响应模型
class AugmentResponseChunk(BaseModel):
    """Augment API响应块"""
    text: str
    unknown_blob_names: List[Any] = []
    checkpoint_not_found: bool = False
    workspace_file_chunks: List[Any] = []
    incorporated_external_sources: List[Any] = []
    nodes: List[AugmentResponseNode] = []

#################################################
# 辅助函数
#################################################

def generate_id():
    """生成唯一ID，类似于OpenAI的格式"""
    return str(uuid.uuid4()).replace("-", "")[:24]

def estimate_tokens(text):
    """
    估计文本的token数量
    这是一个简单的估算，实际数量可能有所不同
    """
    if not text:
        return 0
    # 简单估算：假设每个单词约等于1.3个token
    # 中文字符每个字约等于1个token
    words = len(text.split()) if text else 0
    chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff') if text else 0
    return int(words * 1.3 + chinese_chars)

def map_model_name(openai_model: str) -> Optional[str]:
    """
    将OpenAI模型名称映射到Augment模型名称
    
    Args:
        openai_model: OpenAI格式的模型名称
        
    Returns:
        Augment格式的模型名称，或None表示使用自动选择
    """
    # 模型名称映射规则
    if openai_model == "augment-auto":
        # 使用null表示自动选择模型
        return None
    elif openai_model.startswith("claude-"):
        # Claude模型名称，添加augment-前缀
        return f"augment-{openai_model}"
    elif openai_model.startswith("augment-"):
        # 已经是Augment格式的名称，直接使用
        return openai_model
    else:
        # 其他名称默认使用自动选择
        logger.info(f"未知模型名称 '{openai_model}'，使用自动选择")
        return None

# 新增：处理内容数组的函数
def process_content_array(content_array: List[ContentItem]) -> str:
    """
    将内容数组转换为单个字符串
    
    Args:
        content_array: 内容项数组
        
    Returns:
        合并后的文本内容
    """
    result = ""
    for item in content_array:
        if item.type == "text" and item.text:
            result += item.text
    return result

def convert_to_augment_request(openai_request: ChatCompletionRequest) -> AugmentChatRequest:
    """
    将OpenAI API请求转换为Augment API请求
    
    Args:
        openai_request: OpenAI API请求对象
        
    Returns:
        转换后的Augment API请求对象
        
    Raises:
        HTTPException: 如果请求格式无效
    """
    chat_history = []
    system_message = None
    
    # 预处理所有消息，处理内容数组
    for i in range(len(openai_request.messages)):
        msg = openai_request.messages[i]
        if isinstance(msg.content, list):
            # 将内容数组转换为单个字符串
            openai_request.messages[i].content = process_content_array(msg.content)
    
    # 处理消息历史记录
    for i in range(len(openai_request.messages) - 1):
        msg = openai_request.messages[i]
        if msg.role == "system":
            system_message = msg.content
        elif msg.role == "user" and i + 1 < len(openai_request.messages) and openai_request.messages[i + 1].role == "assistant":
            user_msg = msg.content
            assistant_msg = openai_request.messages[i + 1].content
            
            # 创建历史记录条目，格式符合Augment API
            history_item = AugmentChatHistoryItem(
                request_message=user_msg,
                response_text=assistant_msg,
                request_id=generate_id(),
                response_nodes=[
                    AugmentResponseNode(
                        id=0,
                        type=0,
                        content=assistant_msg,
                        tool_use=None
                    )
                ]
            )
            chat_history.append(history_item)
    
    # 获取当前用户消息
    current_message = None
    for msg in reversed(openai_request.messages):
        if msg.role == "user":
            current_message = msg.content
            break
    
    # 如果没有用户消息，则返回错误
    if current_message is None:
        raise HTTPException(
            status_code=400,
            detail="At least one user message is required"
        )
    
    # 映射模型名称
    augment_model = map_model_name(openai_request.model)
    
    # 准备Augment请求体
    augment_request = AugmentChatRequest(
        model=augment_model,
        message=current_message,
        chat_history=chat_history,
        mode="CHAT"
    )
    
    # 如果有系统消息，设置为用户指南
    if system_message:
        augment_request.user_guidelines = system_message
    
    return augment_request

#################################################
# FastAPI应用
#################################################

def create_app(augment_base_url, chat_endpoint, timeout, max_connections, max_keepalive, keepalive_expiry):
    """
    创建并配置FastAPI应用
    
    Args:
        augment_base_url: Augment API基础URL
        chat_endpoint: 聊天端点路径
        timeout: 请求超时时间
        max_connections: 连接池最大连接数
        max_keepalive: 保持活动的连接数
        keepalive_expiry: 连接保持活动的时间(秒)
        
    Returns:
        配置好的FastAPI应用
    """
    app = FastAPI(
        title="OpenAI to Augment API Adapter",
        description="A FastAPI adapter that converts OpenAI API requests to Augment API format",
        version="1.0.0"
    )

    # 添加CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # HTTP客户端连接池
    http_client = None
    
    @app.on_event("startup")
    async def startup_event():
        """应用启动时初始化HTTP客户端连接池"""
        nonlocal http_client
        http_client = httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive,
                keepalive_expiry=keepalive_expiry
            )
        )
        logger.info(f"已初始化HTTP客户端连接池: 最大连接数={max_connections}, 保持活动连接数={max_keepalive}, 连接过期时间={keepalive_expiry}秒")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """应用关闭时关闭HTTP客户端连接池"""
        nonlocal http_client
        if http_client:
            await http_client.aclose()
            logger.info("已关闭HTTP客户端连接池")

    #################################################
    # 中间件和依赖项
    #################################################

    @app.middleware("http")
    async def catch_exceptions_middleware(request: Request, call_next):
        """捕获所有未处理的异常，返回适当的错误响应"""
        try:
            return await call_next(request)
        except Exception as e:
            logger.exception("Unhandled exception")
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "message": str(e),
                        "type": "internal_server_error",
                        "param": None,
                        "code": "internal_server_error"
                    }
                }
            )

    async def verify_api_key(authorization: str = Header(...)):
        """
        验证API密钥
        
        Args:
            authorization: Authorization头部值
            
        Returns:
            提取的API密钥
            
        Raises:
            HTTPException: 如果API密钥格式无效或为空
        """
        if not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=401, 
                detail={
                    "error": {
                        "message": "Invalid API key format. Expected 'Bearer YOUR_API_KEY'",
                        "type": "invalid_request_error",
                        "param": "authorization",
                        "code": "invalid_api_key"
                    }
                }
            )
        api_key = authorization.replace("Bearer ", "")
        if not api_key:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "message": "API key cannot be empty",
                        "type": "invalid_request_error",
                        "param": "authorization",
                        "code": "invalid_api_key"
                    }
                }
            )
        return api_key

    #################################################
    # API端点
    #################################################

    @app.get("/health")
    async def health_check():
        """健康检查端点"""
        return {"status": "ok", "timestamp": datetime.now().isoformat()}

    @app.get("/v1/models")
    async def list_models():
        """列出支持的模型"""
        # 返回支持的模型列表，包含Augment支持的模型
        models = [
            ModelInfo(id="augment-auto", created=int(time.time())),
            ModelInfo(id="claude-3.7-sonnet", created=int(time.time())),
            ModelInfo(id="augment-claude-3.7-sonnet", created=int(time.time())),
        ]
        return ModelListResponse(data=models)

    @app.get("/v1/models/{model_id}")
    async def get_model(model_id: str):
        """获取特定模型的信息"""
        return ModelInfo(id=model_id, created=int(time.time()))

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: ChatCompletionRequest,
        api_key: str = Depends(verify_api_key)
    ):
        """
        聊天完成端点 - 将OpenAI API请求转换为Augment API请求
        
        Args:
            request: OpenAI格式的聊天完成请求
            api_key: 通过验证的API密钥
            
        Returns:
            OpenAI格式的聊天完成响应或流式响应
        """
        try:
            # 转换为Augment请求格式
            augment_request = convert_to_augment_request(request)
            logger.debug(f"Converted request: {augment_request.model_dump(exclude_none=True)}")
            
            # 决定是否使用流式响应
            if request.stream:
                return StreamingResponse(
                    stream_augment_response(http_client, augment_base_url, api_key, augment_request, request.model, chat_endpoint),
                    media_type="text/event-stream"
                )
            else:
                # 同步请求处理
                return await handle_sync_request(http_client, augment_base_url, api_key, augment_request, request.model, chat_endpoint)
                
        except httpx.TimeoutException:
            logger.error("Request to Augment API timed out")
            raise HTTPException(
                status_code=504,
                detail={
                    "error": {
                        "message": "Request to Augment API timed out",
                        "type": "timeout_error",
                        "param": None,
                        "code": "timeout"
                    }
                }
            )
        except httpx.HTTPError as e:
            logger.error(f"HTTP error: {str(e)}")
            raise HTTPException(
                status_code=502,
                detail={
                    "error": {
                        "message": f"Error communicating with Augment API: {str(e)}",
                        "type": "api_error",
                        "param": None,
                        "code": "api_error"
                    }
                }
            )
        except HTTPException:
            # 重新抛出HTTPException，以保持原始状态码和详细信息
            raise
        except Exception as e:
            logger.exception("Unexpected error")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": {
                        "message": f"Internal server error: {str(e)}",
                        "type": "internal_server_error",
                        "param": None,
                        "code": "internal_server_error"
                    }
                }
            )

    return app

async def handle_sync_request(client, base_url, api_key, augment_request, model_name, chat_endpoint):
    """
    处理同步请求
    
    Args:
        client: HTTP客户端连接池
        base_url: Augment API基础URL
        api_key: API密钥
        augment_request: Augment API请求对象
        model_name: 模型名称
        chat_endpoint: 聊天端点
        
    Returns:
        OpenAI格式的聊天完成响应
    """
    # 排除None值，确保正确的JSON格式
    request_json = augment_request.model_dump(exclude_none=True)
    
    response = await client.post(
        f"{base_url.rstrip('/')}/{chat_endpoint}",
        json=request_json,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "Augment.openai-adapter/1.0.0",
            "Accept": "*/*"
        }
    )
    
    if response.status_code != 200:
        logger.error(f"Augment API error: {response.status_code} - {response.text}")
        raise HTTPException(
            status_code=response.status_code,
            detail={
                "error": {
                    "message": f"Augment API error: {response.text}",
                    "type": "api_error",
                    "param": None,
                    "code": "api_error"
                }
            }
        )
    
    # 处理流式响应，合并为完整响应
    full_response = ""
    for line in response.text.split("\n"):
        if line.strip():
            try:
                data = json.loads(line)
                if "text" in data and data["text"]:
                    full_response += data["text"]
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON: {line}")
    
    # 估算token使用情况
    prompt_tokens = estimate_tokens(augment_request.message)
    completion_tokens = estimate_tokens(full_response)
    
    # 构建OpenAI格式响应
    return ChatCompletionResponse(
        id=f"chatcmpl-{generate_id()}",
        created=int(time.time()),
        model=model_name,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content=full_response
                ),
                finish_reason="stop"
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
    )

async def stream_augment_response(client, base_url, api_key, augment_request, model_name, chat_endpoint):
    """
    处理流式响应
    
    Args:
        client: HTTP客户端连接池
        base_url: Augment API基础URL
        api_key: API密钥
        augment_request: Augment API请求对象
        model_name: 模型名称
        chat_endpoint: 聊天端点
        
    Yields:
        流式响应的数据块
    """
    try:
        # 排除None值，确保正确的JSON格式
        request_json = augment_request.model_dump(exclude_none=True)
        
        async with client.stream(
            "POST",
            f"{base_url.rstrip('/')}/{chat_endpoint}",
            json=request_json,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "Augment.openai-adapter/1.0.0",
                "Accept": "*/*"
            }
        ) as response:
            
            if response.status_code != 200:
                error_detail = await response.aread()
                logger.error(f"Augment API error: {response.status_code} - {error_detail}")
                error_message = f"Error from Augment API: {error_detail.decode('utf-8', errors='replace')}"
                yield f"data: {json.dumps({'error': error_message})}\n\n"
                return
            
            # 生成唯一ID
            chat_id = f"chatcmpl-{generate_id()}"
            created_time = int(time.time())
            
            # 初始化响应
            init_response = ChatCompletionStreamResponse(
                id=chat_id,
                created=created_time,
                model=model_name,
                choices=[
                    ChatCompletionStreamResponseChoice(
                        index=0,
                        delta={"role": "assistant"},
                        finish_reason=None
                    )
                ]
            )
            init_data = json.dumps(init_response.model_dump())
            yield f"data: {init_data}\n\n"
            
            # 处理流式响应
            buffer = ""
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                
                try:
                    # 解析Augment响应格式
                    chunk = json.loads(line)
                    if "text" in chunk and chunk["text"]:
                        content = chunk["text"]
                        
                        # 发送增量更新
                        stream_response = ChatCompletionStreamResponse(
                            id=chat_id,
                            created=created_time,
                            model=model_name,
                            choices=[
                                ChatCompletionStreamResponseChoice(
                                    index=0,
                                    delta={"content": content},
                                    finish_reason=None
                                )
                            ]
                        )
                        response_data = json.dumps(stream_response.model_dump())
                        yield f"data: {response_data}\n\n"
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON: {line}")
            
            # 发送完成信号
            final_response = ChatCompletionStreamResponse(
                id=chat_id,
                created=created_time,
                model=model_name,
                choices=[
                    ChatCompletionStreamResponseChoice(
                        index=0,
                        delta={},
                        finish_reason="stop"
                    )
                ]
            )
            final_data = json.dumps(final_response.model_dump())
            yield f"data: {final_data}\n\n"
            
            # 发送[DONE]标记
            yield "data: [DONE]\n\n"
            
    except httpx.TimeoutException:
        logger.error("Request to Augment API timed out")
        yield f"data: {json.dumps({'error': 'Request to Augment API timed out'})}\n\n"
    except httpx.HTTPError as e:
        logger.error(f"HTTP error: {str(e)}")
        yield f"data: {json.dumps({'error': f'Error communicating with Augment API: {str(e)}'})}\n\n"
    except Exception as e:
        logger.exception("Unexpected error")
        yield f"data: {json.dumps({'error': f'Internal server error: {str(e)}'})}\n\n"

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="OpenAI to Augment API Adapter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--augment-url", 
        default="https://d18.api.augmentcode.com/",
        help="Augment API基础URL"
    )
    
    parser.add_argument(
        "--chat-endpoint", 
        default="chat-stream",
        help="Augment聊天端点路径"
    )
    
    parser.add_argument(
        "--host", 
        default="0.0.0.0",
        help="服务器主机地址"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=8686,
        help="服务器端口"
    )
    
    parser.add_argument(
        "--timeout", 
        type=int, 
        default=120,
        help="API请求超时时间（秒）"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="启用调试模式"
    )

    parser.add_argument(
        "--tenant-id",
        default="d18",
        help="Augment API租户ID (域名前缀)"
    )
    
    # 连接池相关参数
    parser.add_argument(
        "--max-connections",
        type=int,
        default=100,
        help="HTTP连接池最大连接数"
    )
    
    parser.add_argument(
        "--max-keepalive",
        type=int,
        default=20,
        help="HTTP连接池保持活动的连接数"
    )
    
    parser.add_argument(
        "--keepalive-expiry",
        type=float,
        default=60.0,
        help="HTTP连接池连接保持活动的时间(秒)"
    )
    
    return parser.parse_args()

#################################################
# 主程序
#################################################

def main():
    """主函数"""
    args = parse_args()
    
    # 配置日志级别
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 构建完整的Augment URL
    if args.augment_url == "https://d18.api.augmentcode.com/":
        # 如果使用默认URL，则应用tenant-id参数
        augment_base_url = f"https://{args.tenant_id}.api.augmentcode.com/"
        logger.info(f"Using tenant ID: {args.tenant_id}")
    else:
        # 否则使用提供的URL
        augment_base_url = args.augment_url
    
    # 创建应用
    app = create_app(
        augment_base_url=augment_base_url,
        chat_endpoint=args.chat_endpoint,
        timeout=args.timeout,
        max_connections=args.max_connections,
        max_keepalive=args.max_keepalive,
        keepalive_expiry=args.keepalive_expiry
    )
    
    # 启动应用
    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info(f"Using Augment base URL: {augment_base_url}")
    logger.info(f"Using Augment chat endpoint: {args.chat_endpoint}")
    logger.info(f"HTTP连接池配置: 最大连接数={args.max_connections}, 保持活动连接数={args.max_keepalive}, 连接过期时间={args.keepalive_expiry}秒")
    
    uvicorn.run(
        app, 
        host=args.host, 
        port=args.port,
        log_level="info" if not args.debug else "debug"
    )

if __name__ == "__main__":
    main()