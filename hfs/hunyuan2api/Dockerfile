# --- 第一阶段：构建阶段 (Builder Stage) ---
# 使用官方的 Go 语言镜像作为编译环境， Alpine 版本比较小巧
FROM golang:1.22-alpine AS builder
# 或者 FROM golang:1.22 # 如果 alpine 的 musl libc 与您的代码有兼容问题

# 设置构建阶段的工作目录
WORKDIR /build

# 将你的 Go 源代码文件 (hunyuan2api.go) 复制到构建环境的 /build/ 目录下
COPY hunyuan2api.go .

# 编译 Go 应用程序
# CGO_ENABLED=0 尝试进行静态链接，避免 C 库依赖问题，尤其是在使用 alpine 镜像时
# -ldflags="-w -s" 用于减小编译后二进制文件的大小
# -o /app/hunyuan2api 指定编译输出的可执行文件路径和名称
# hunyuan2api.go 是你的源文件名
RUN CGO_ENABLED=0 go build -ldflags="-w -s" -o /app/hunyuan2api hunyuan2api.go

# --- 第二阶段：运行阶段 (Final Stage) ---
# 使用一个非常精简的基础镜像来运行编译好的程序
FROM alpine:latest
# 注意：如果静态编译 (CGO_ENABLED=0) 失败或运行时仍有问题，
# 可能需要换成基于 glibc 的镜像，例如 'debian:stable-slim'
# FROM debian:stable-slim

# 设置最终运行阶段的工作目录
WORKDIR /app

# 从第一阶段 (builder) 复制编译好的二进制文件到最终镜像的 /app/ 目录下
COPY --from=builder /app/hunyuan2api /app/hunyuan2api

# 确保复制过来的二进制文件具有执行权限
RUN chmod +x /app/hunyuan2api

# 暴露你的 Go 应用程序监听的网络端口 (根据你的启动参数是 6677)
EXPOSE 6677

# 设置容器启动时执行的命令
# 这里的启动参数需要和您提供的一致
CMD ["/app/hunyuan2api", "--address", "0.0.0.0", "--port", "6677", "--verify-ssl=false", "--dev", "--workers", "400", "--queue-size", "1000", "--max-concurrent", "400"]