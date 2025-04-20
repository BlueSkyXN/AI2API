FROM node:16-slim

# 安装 git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 克隆代码仓库
RUN git clone https://github.com/Rfym21/Qwen2API .

# 预先创建数据目录并设置权限
RUN mkdir -p /app/data && \
    chmod 777 /app/data && \
    chmod 777 /app

# 安装依赖
RUN npm install

# 暴露端口
EXPOSE 3000

# 创建不写入 .env 文件的启动脚本
RUN echo '#!/bin/bash\n\
\n\
# 日志函数\n\
log() {\n\
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] $1"\n\
}\n\
\n\
# 显示配置信息\n\
log "配置信息:"\n\
log "API_PREFIX: ${API_PREFIX:-(未设置)}"\n\
log "SERVICE_PORT: ${SERVICE_PORT:-3000}"\n\
log "API_KEY: ${API_KEY:+已设置} ${API_KEY:-未设置}"\n\
log "ACCOUNT_TOKENS: ${ACCOUNT_TOKENS:+已设置} ${ACCOUNT_TOKENS:-未设置}"\n\
log "SEARCH_INFO_MODE: ${SEARCH_INFO_MODE:-table}"\n\
\n\
# 直接使用环境变量启动服务\n\
log "正在启动 Qwen2API 服务..."\n\
node src/server.js\n\
' > /app/start.sh && chmod +x /app/start.sh

# 设置启动命令
CMD ["/app/start.sh"]