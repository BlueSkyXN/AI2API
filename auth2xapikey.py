from fastapi import FastAPI, Request, Header
import httpx
import uvicorn
from typing import Optional
from fastapi.responses import StreamingResponse, Response
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

TARGET_API_URL = "https://rad.huddlz.xyz"  # 替换为您的目标API地址

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"])
async def proxy_request(request: Request, path: str, authorization: Optional[str] = Header(None)):
    # 记录请求信息
    logger.info(f"接收到请求: {request.method} {request.url.path}")
    
    # 获取原始请求体
    body = await request.body()
    
    # 获取并处理原始请求头
    headers = dict(request.headers)
    logger.debug(f"原始请求头: {headers}")
    
    # 从Authorization中提取token并设置为x-api-key
    if authorization:
        token = authorization.replace("Bearer ", "")
        headers["x-api-key"] = token
        logger.debug(f"从Authorization提取并设置x-api-key")
    
    # 移除可能导致问题的请求头
    headers.pop("host", None)
    
    # 处理内容编码相关头信息
    # 移除Accept-Encoding头，让httpx自己处理内容压缩/解压缩
    if "accept-encoding" in headers:
        logger.debug(f"移除Accept-Encoding: {headers.get('accept-encoding')}")
        headers.pop("accept-encoding", None)
    
    # 构建完整的目标URL
    url = f"{TARGET_API_URL}/{path}"
    logger.info(f"转发请求到: {url}")
    
    # 获取查询参数
    params = dict(request.query_params)
    
    try:
        # 转发请求到目标API，禁用自动处理压缩内容
        async with httpx.AsyncClient(headers={"Accept-Encoding": "identity"}) as client:
            response = await client.request(
                method=request.method,
                url=url,
                params=params,
                headers=headers,
                content=body,
                timeout=60.0,
                follow_redirects=True
            )
            
            logger.info(f"收到响应: 状态码 {response.status_code}")
            logger.debug(f"响应头: {response.headers}")
            
            # 处理响应头，移除可能导致问题的头信息
            response_headers = dict(response.headers)
            
            # 移除可能导致冲突的头信息
            headers_to_remove = [
                "content-length", 
                "transfer-encoding", 
                "content-encoding",  # 重要：移除内容编码头
                "server", 
                "connection"
            ]
            
            for header in headers_to_remove:
                if header in response_headers:
                    logger.debug(f"移除响应头: {header}")
                    response_headers.pop(header, None)
            
            # 获取响应内容（已自动解压缩）
            content = await response.aread()
            
            # 返回未压缩的响应
            return Response(
                content=content,
                status_code=response.status_code,
                headers=response_headers
            )
    except Exception as e:
        logger.error(f"请求处理过程中发生错误: {str(e)}", exc_info=True)
        return {"detail": f"代理服务器错误: {str(e)}"}, 500

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9898) 