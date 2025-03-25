import base64
import hashlib
import json
import os
import secrets
import urllib.parse
from typing import Dict, Any

import requests


def base64url_encode(data: bytes) -> str:
    """将数据进行 base64url 编码"""
    return base64.urlsafe_b64encode(data).decode('utf-8').replace('=', '')


def sha256_hash(input_data: bytes) -> bytes:
    """计算输入数据的 SHA-256 哈希值"""
    return hashlib.sha256(input_data).digest()


def create_oauth_state() -> Dict[str, Any]:
    """创建 OAuth 状态对象"""
    code_verifier_bytes = secrets.token_bytes(32)
    code_verifier = base64url_encode(code_verifier_bytes)
    
    code_challenge_bytes = sha256_hash(code_verifier.encode('utf-8'))
    code_challenge = base64url_encode(code_challenge_bytes)
    
    state = base64url_encode(secrets.token_bytes(8))
    
    oauth_state = {
        "codeVerifier": code_verifier,
        "codeChallenge": code_challenge,
        "state": state,
        "creationTime": int(import_time())
    }
    
    return oauth_state


def generate_authorize_url(oauth_state: Dict[str, Any]) -> str:
    """生成授权 URL"""
    client_id = "v"
    
    params = {
        "response_type": "code",
        "code_challenge": oauth_state["codeChallenge"],
        "client_id": client_id,
        "state": oauth_state["state"],
        "prompt": "login"
    }
    
    query_string = urllib.parse.urlencode(params)
    authorize_url = f"https://auth.augmentcode.com/authorize?{query_string}"
    
    return authorize_url


def get_access_token(tenant_url: str, code_verifier: str, code: str) -> str:
    """获取访问令牌"""
    data = {
        "grant_type": "authorization_code",
        "client_id": "v",
        "code_verifier": code_verifier,
        "redirect_uri": "",
        "code": code
    }
    
    response = requests.post(
        f"{tenant_url}token",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    json_response = response.json()
    token = json_response.get("access_token")
    
    return token


def parse_code(code_str: str) -> Dict[str, str]:
    """解析返回的代码"""
    parsed = json.loads(code_str)
    return {
        "code": parsed.get("code"),
        "state": parsed.get("state"),
        "tenant_url": parsed.get("tenant_url")
    }


def import_time():
    """获取当前时间戳"""
    import time
    return time.time() * 1000


def main():
    """主函数"""
    print("正在生成 OAuth 状态...")
    oauth_state = create_oauth_state()
    print("OAuth 状态已生成:")
    print(json.dumps(oauth_state, indent=2))
    
    url = generate_authorize_url(oauth_state)
    print("\n请访问以下 URL 进行授权:")
    print(url)
    
    code_str = input("\n请输入返回的代码 (JSON 格式): ")
    
    try:
        parsed_code = parse_code(code_str)
        print("代码已解析:")
        print(json.dumps(parsed_code, indent=2))
        
        print("\n正在获取访问令牌...")
        token = get_access_token(
            parsed_code["tenant_url"],
            oauth_state["codeVerifier"],
            parsed_code["code"]
        )
        
        print("\n访问令牌:")
        print(token)
        
    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == "__main__":
    main()