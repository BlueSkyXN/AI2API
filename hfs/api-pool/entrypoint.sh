#!/bin/sh
# 如果命令失败则立即退出
set -e

# 定义临时密钥文件的路径
KEY_FILE_PATH="/tmp/keys.txt"

echo "--- 正在检查 Secrets ---"

# 检查 API_PASSWORD 是否已设置
if [ -z "${API_PASSWORD}" ]; then
  echo "[错误] 必须在 Hugging Face Secrets 中设置 API_PASSWORD !"
  exit 1
fi

# 检查 key_list 是否已设置
if [ -z "${key_list}" ]; then
  echo "[错误] 必须在 Hugging Face Secrets 中设置 key_list !"
  exit 1
fi

echo "--- 正在从 Secret 'key_list' 创建临时密钥文件 (${KEY_FILE_PATH}) ---"

# 从环境变量 key_list 读取内容，并写入临时文件
# 使用 'echo -e' 来解释可能存在的 '\n' 换行符
# 将标准错误重定向到 /dev/null 以避免打印潜在的密钥内容（尽管通常 echo 不会）
echo -e "${key_list}" > "${KEY_FILE_PATH}" 2>/dev/null

# 验证文件是否创建成功且非空
if [ ! -s "${KEY_FILE_PATH}" ]; then
    echo "[错误] 创建密钥文件失败或文件为空！请检查 'key_list' Secret 的内容。"
    exit 1
fi

echo "--- 密钥文件已生成 ---"

# !!! 【重要】生产环境中不要取消下面这行的注释，避免日志泄露 !!!
# echo "密钥文件内容预览 (前几行):"
# head -n 3 "${KEY_FILE_PATH}"

echo "--- 正在启动 api-pool 服务 ---"

# 使用 exec 执行 Go 程序，将脚本进程替换为 Go 程序进程
# 将临时文件路径传给 --key-file
# 将从 Secret 读取的密码传给 --password
# 将地址设为 0.0.0.0 以便容器外访问
# 传入其他您指定的参数
exec /app/api-pool \
    --key-file "${KEY_FILE_PATH}" \
    --target-url "https://api.siliconflow.cn" \
    --port "6969" \
    --address "0.0.0.0" \
    --password "${API_PASSWORD}" \
    --max-workers=1000 \
    --max-queue=2000
    # 注意：--max-workers 和 --max-queue 值较高，请关注 Space 资源使用情况