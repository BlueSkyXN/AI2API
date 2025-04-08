# --- 第一阶段：构建阶段 (Builder Stage) ---
# 使用官方 Go 镜像进行编译
FROM golang:1.22-alpine AS builder

# 设置工作目录
WORKDIR /build

# 复制 Go 源代码文件
COPY api-pool.go .

# 编译 Go 应用
# CGO_ENABLED=0 尝试静态链接，减少依赖
# -ldflags="-w -s" 减小二进制文件大小
# -o /app/api-pool 指定输出路径和名称
RUN CGO_ENABLED=0 go build -ldflags="-w -s" -o /app/api-pool api-pool.go

# --- 第二阶段：运行阶段 (Final Stage) ---
# 使用轻量的 Alpine 镜像作为最终运行环境
FROM alpine:latest

# 设置工作目录
WORKDIR /app

# 从构建阶段复制编译好的二进制文件
COPY --from=builder /app/api-pool /app/api-pool

# 复制启动脚本
COPY entrypoint.sh /app/entrypoint.sh

# 赋予执行权限
RUN chmod +x /app/api-pool /app/entrypoint.sh

# 暴露应用程序监听的端口 (根据您的参数是 6969)
EXPOSE 6969

# 设置容器的入口点为启动脚本
ENTRYPOINT ["/app/entrypoint.sh"]

# 注意：CMD 指令现在由 entrypoint.sh 脚本通过 exec 来执行