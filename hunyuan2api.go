package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

// 配置结构体用于存储命令行参数
type Config struct {
	Port         string // 代理服务器监听端口
	Address      string // 代理服务器监听地址
	LogLevel     string // 日志级别
	DevMode      bool   // 开发模式标志
	MaxRetries   int    // 最大重试次数
	Timeout      int    // 请求超时时间(秒)
	VerifySSL    bool   // 是否验证SSL证书
	ModelName    string // 默认模型名称
	BearerToken  string // Bearer Token (默认提供公开Token)
}

// 支持的模型列表
var SupportedModels = []string{
	"hunyuan-t1-latest",
	"hunyuan-turbos-latest",
}

// 腾讯混元 API 目标URL
const (
	TargetURL = "https://llm.hunyuan.tencent.com/aide/api/v2/triton_image/demo_text_chat/"
	Version   = "1.0.0" // 版本号
)

// 日志级别
const (
	LogLevelDebug = "debug"
	LogLevelInfo  = "info"
	LogLevelWarn  = "warn"
	LogLevelError = "error"
)

// 解析命令行参数并返回 Config 实例
func parseFlags() *Config {
	cfg := &Config{}
	flag.StringVar(&cfg.Port, "port", "6666", "Port to listen on")
	flag.StringVar(&cfg.Address, "address", "localhost", "Address to listen on")
	flag.StringVar(&cfg.LogLevel, "log-level", LogLevelInfo, "Log level (debug, info, warn, error)")
	flag.BoolVar(&cfg.DevMode, "dev", false, "Enable development mode with enhanced logging")
	flag.IntVar(&cfg.MaxRetries, "max-retries", 3, "Maximum number of retries for failed requests")
	flag.IntVar(&cfg.Timeout, "timeout", 300, "Request timeout in seconds")
	flag.BoolVar(&cfg.VerifySSL, "verify-ssl", true, "Verify SSL certificates")
	flag.StringVar(&cfg.ModelName, "model", "hunyuan-t1-latest", "Default Hunyuan model name")
	flag.StringVar(&cfg.BearerToken, "token", "7auGXNATFSKl7dF", "Bearer token for Hunyuan API")
	flag.Parse()
	
	// 如果开发模式开启，自动设置日志级别为debug
	if cfg.DevMode && cfg.LogLevel != LogLevelDebug {
		cfg.LogLevel = LogLevelDebug
		fmt.Println("开发模式已启用，日志级别设置为debug")
	}
	
	return cfg
}

// 全局配置变量
var (
	appConfig *Config
)

// 性能指标
var (
	requestCounter    int64
	successCounter    int64
	errorCounter      int64
	avgResponseTime   int64
	latencyHistogram  [10]int64 // 0-100ms, 100-200ms, ... >1s
)

// 日志记录器
var (
	logger    *log.Logger
	logLevel  string
	logMutex  sync.Mutex
)

// 日志初始化
func initLogger(level string) {
	logger = log.New(os.Stdout, "[HunyuanAPI] ", log.LstdFlags)
	logLevel = level
}

// 根据日志级别记录日志
func logDebug(format string, v ...interface{}) {
	if logLevel == LogLevelDebug {
		logMutex.Lock()
		logger.Printf("[DEBUG] "+format, v...)
		logMutex.Unlock()
	}
}

func logInfo(format string, v ...interface{}) {
	if logLevel == LogLevelDebug || logLevel == LogLevelInfo {
		logMutex.Lock()
		logger.Printf("[INFO] "+format, v...)
		logMutex.Unlock()
	}
}

func logWarn(format string, v ...interface{}) {
	if logLevel == LogLevelDebug || logLevel == LogLevelInfo || logLevel == LogLevelWarn {
		logMutex.Lock()
		logger.Printf("[WARN] "+format, v...)
		logMutex.Unlock()
	}
}

func logError(format string, v ...interface{}) {
	logMutex.Lock()
	logger.Printf("[ERROR] "+format, v...)
	logMutex.Unlock()
	
	// 错误计数
	atomic.AddInt64(&errorCounter, 1)
}

// OpenAI/DeepSeek 消息格式
type APIMessage struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"` // 使用interface{}以支持各种类型
}

// OpenAI/DeepSeek 请求格式
type APIRequest struct {
	Model       string       `json:"model"`
	Messages    []APIMessage `json:"messages"`
	Stream      bool         `json:"stream"`
	Temperature float64      `json:"temperature,omitempty"`
	MaxTokens   int          `json:"max_tokens,omitempty"`
}

// 腾讯混元请求格式
type HunyuanRequest struct {
	Stream           bool         `json:"stream"`
	Model            string       `json:"model"`
	QueryID          string       `json:"query_id"`
	Messages         []APIMessage `json:"messages"`
	StreamModeration bool         `json:"stream_moderation"`
	EnableEnhancement bool        `json:"enable_enhancement"`
}

// 腾讯混元响应格式
type HunyuanResponse struct {
	ID                string      `json:"id"`
	Object            string      `json:"object"`
	Created           int64       `json:"created"`
	Model             string      `json:"model"`
	SystemFingerprint string      `json:"system_fingerprint"`
	Choices           []Choice    `json:"choices"`
	Note              string      `json:"note,omitempty"`
}

// 选择结构
type Choice struct {
	Index        int     `json:"index"`
	Delta        Delta   `json:"delta"`
	FinishReason *string `json:"finish_reason"`
}

// Delta结构，包含内容和推理内容
type Delta struct {
	Role             string `json:"role,omitempty"`
	Content          string `json:"content,omitempty"`
	ReasoningContent string `json:"reasoning_content,omitempty"`
}

// DeepSeek 流式响应格式
type StreamChunk struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index        int     `json:"index"`
		FinishReason *string `json:"finish_reason,omitempty"`
		Delta        struct {
			Role             string `json:"role,omitempty"`
			Content          string `json:"content,omitempty"`
			ReasoningContent string `json:"reasoning_content,omitempty"`
		} `json:"delta"`
	} `json:"choices"`
}

// 非流式响应格式
type CompletionResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index        int    `json:"index"`
		FinishReason string `json:"finish_reason"`
		Message      struct {
			Role             string `json:"role"`
			Content          string `json:"content"`
			ReasoningContent string `json:"reasoning_content,omitempty"`
		} `json:"message"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

// 请求计数和互斥锁，用于监控
var (
	requestCount uint64 = 0
	countMutex   sync.Mutex
)

// 主入口函数
func main() {
	// 解析配置
	appConfig = parseFlags()
	
	// 初始化日志
	initLogger(appConfig.LogLevel)

	logInfo("启动服务: TargetURL=%s, Address=%s, Port=%s, Version=%s, LogLevel=%s, 支持模型=%v, BearerToken=***",
		TargetURL, appConfig.Address, appConfig.Port, Version, appConfig.LogLevel, SupportedModels)

	// 配置更高的并发处理能力
	http.DefaultTransport.(*http.Transport).MaxIdleConnsPerHost = 100
	http.DefaultTransport.(*http.Transport).MaxIdleConns = 100
	http.DefaultTransport.(*http.Transport).IdleConnTimeout = 90 * time.Second
	
	// 创建自定义服务器，支持更高并发
	server := &http.Server{
		Addr:         appConfig.Address + ":" + appConfig.Port,
		ReadTimeout:  time.Duration(appConfig.Timeout) * time.Second,
		WriteTimeout: time.Duration(appConfig.Timeout) * time.Second,
		IdleTimeout:  120 * time.Second,
		Handler:      nil, // 使用默认的ServeMux
	}

	// 创建处理器
	http.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		setCORSHeaders(w)
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}
		handleModelsRequest(w, r)
	})

	http.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		setCORSHeaders(w)
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}
		
		// 设置超时上下文
		ctx, cancel := context.WithTimeout(r.Context(), time.Duration(appConfig.Timeout)*time.Second)
		defer cancel()
		
		// 包含超时上下文的请求
		r = r.WithContext(ctx)
		
		// 添加恢复机制，防止panic
		defer func() {
			if r := recover(); r != nil {
				logError("处理请求时发生panic: %v", r)
				http.Error(w, "Internal server error", http.StatusInternalServerError)
			}
		}()
		
		// 计数器增加
		countMutex.Lock()
		requestCount++
		currentCount := requestCount
		countMutex.Unlock()
		
		logInfo("收到新请求 #%d", currentCount)
		
		// 请求计数
		atomic.AddInt64(&requestCounter, 1)
		
		// 处理请求
		handleChatCompletionRequest(w, r)
	})
	
	// 添加健康检查端点
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		setCORSHeaders(w)
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}
		
		countMutex.Lock()
		count := requestCount
		countMutex.Unlock()
		
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(fmt.Sprintf(`{"status":"ok","version":"%s","requests":%d}`, Version, count)))
	})
	
	// 创建停止通道
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)
	
	// 在goroutine中启动服务器
	go func() {
		logInfo("Starting proxy server on %s", server.Addr)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logError("Failed to start server: %v", err)
			os.Exit(1)
		}
	}()
	
	// 等待停止信号
	<-stop
	
	// 创建上下文用于优雅关闭
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	
	// 优雅关闭服务器
	logInfo("Server is shutting down...")
	if err := server.Shutdown(ctx); err != nil {
		logError("Server shutdown failed: %v", err)
	}
	
	logInfo("Server gracefully stopped")
}

// 设置CORS头
func setCORSHeaders(w http.ResponseWriter) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
}

// 验证消息格式
func validateMessages(messages []APIMessage) (bool, string) {
	reqID := generateRequestID()
	logDebug("[reqID:%s] 验证消息格式", reqID)

	if messages == nil || len(messages) == 0 {
		return false, "Messages array is required"
	}

	for _, msg := range messages {
		if msg.Role == "" || msg.Content == nil {
			return false, "Invalid message format: each message must have role and content"
		}
	}

	return true, ""
}

// 从请求头中提取令牌
func extractToken(r *http.Request) (string, error) {
	// 获取 Authorization 头部
	authHeader := r.Header.Get("Authorization")
	if authHeader == "" {
		return "", fmt.Errorf("missing Authorization header")
	}

	// 验证格式并提取令牌
	if !strings.HasPrefix(authHeader, "Bearer ") {
		return "", fmt.Errorf("invalid Authorization header format, must start with 'Bearer '")
	}

	// 提取令牌值
	token := strings.TrimPrefix(authHeader, "Bearer ")
	if token == "" {
		return "", fmt.Errorf("empty token in Authorization header")
	}

	return token, nil
}

// 转换任意类型的内容为字符串
func contentToString(content interface{}) string {
	if content == nil {
		return ""
	}

	switch v := content.(type) {
	case string:
		return v
	default:
		jsonBytes, err := json.Marshal(v)
		if err != nil {
			logWarn("将内容转换为JSON失败: %v", err)
			return ""
		}
		return string(jsonBytes)
	}
}

// 生成请求ID
func generateQueryID() string {
	return fmt.Sprintf("%s%d", getRandomString(8), time.Now().UnixNano())
}

// 判断模型是否在支持列表中
func isModelSupported(modelName string) bool {
	for _, supportedModel := range SupportedModels {
		if modelName == supportedModel {
			return true
		}
	}
	return false
}

// 处理模型列表请求
func handleModelsRequest(w http.ResponseWriter, r *http.Request) {
	logInfo("处理模型列表请求")

	// 返回模型列表
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)

	// 构建模型数据
	modelData := make([]map[string]interface{}, 0, len(SupportedModels))
	for _, model := range SupportedModels {
		modelData = append(modelData, map[string]interface{}{
			"id":       model,
			"object":   "model",
			"created":  time.Now().Unix(),
			"owned_by": "TencentCloud",
			"capabilities": map[string]interface{}{
				"chat":        true,
				"completions": true,
				"reasoning":   true,
			},
		})
	}

	modelsList := map[string]interface{}{
		"object": "list",
		"data":   modelData,
	}

	json.NewEncoder(w).Encode(modelsList)
}

// 处理聊天补全请求
func handleChatCompletionRequest(w http.ResponseWriter, r *http.Request) {
	reqID := generateRequestID()
	startTime := time.Now()
	logInfo("[reqID:%s] 处理聊天补全请求", reqID)

	// 解析请求体
	var apiReq APIRequest
	if err := json.NewDecoder(r.Body).Decode(&apiReq); err != nil {
		logError("[reqID:%s] 解析请求失败: %v", reqID, err)
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// 验证消息格式
	valid, errMsg := validateMessages(apiReq.Messages)
	if !valid {
		logError("[reqID:%s] 消息格式验证失败: %s", reqID, errMsg)
		http.Error(w, errMsg, http.StatusBadRequest)
		return
	}

	// 是否使用流式处理
	isStream := apiReq.Stream

	// 确定使用的模型
	modelName := appConfig.ModelName
	if apiReq.Model != "" {
		// 检查请求的模型是否是我们支持的
		if isModelSupported(apiReq.Model) {
			modelName = apiReq.Model
		} else {
			logWarn("[reqID:%s] 请求的模型 %s 不支持，使用默认模型 %s", reqID, apiReq.Model, modelName)
		}
	}
	
	logInfo("[reqID:%s] 使用模型: %s", reqID, modelName)

	// 创建混元API请求
	hunyuanReq := HunyuanRequest{
		Stream:            true, // 混元API总是使用流式响应
		Model:             modelName,
		QueryID:           generateQueryID(),
		Messages:          apiReq.Messages,
		StreamModeration:  true,
		EnableEnhancement: false,
	}

	// 转发请求到混元API
	var responseErr error
	if isStream {
		responseErr = handleStreamingRequest(w, r, hunyuanReq, reqID)
	} else {
		responseErr = handleNonStreamingRequest(w, r, hunyuanReq, reqID)
	}
	
	// 请求处理完成，更新指标
	elapsed := time.Since(startTime).Milliseconds()
	
	// 更新延迟直方图
	bucketIndex := min(int(elapsed/100), 9)
	atomic.AddInt64(&latencyHistogram[bucketIndex], 1)
	
	// 更新平均响应时间
	atomic.AddInt64(&avgResponseTime, elapsed)
	
	if responseErr == nil {
		// 成功计数增加
		atomic.AddInt64(&successCounter, 1)
		logInfo("[reqID:%s] 请求处理成功，耗时: %dms", reqID, elapsed)
	} else {
		logError("[reqID:%s] 请求处理失败: %v, 耗时: %dms", reqID, responseErr, elapsed)
	}
}

// 安全的HTTP客户端，支持禁用SSL验证
func getHTTPClient() *http.Client {
	tr := &http.Transport{
		MaxIdleConnsPerHost: 100,
		IdleConnTimeout:     90 * time.Second,
		TLSClientConfig:     nil, // 默认配置
	}

	// 如果配置了禁用SSL验证
	if !appConfig.VerifySSL {
		tr.TLSClientConfig = &tls.Config{InsecureSkipVerify: true}
	}

	return &http.Client{
		Timeout:   time.Duration(appConfig.Timeout) * time.Second,
		Transport: tr,
	}
}

// 处理流式请求
func handleStreamingRequest(w http.ResponseWriter, r *http.Request, hunyuanReq HunyuanRequest, reqID string) error {
	logInfo("[reqID:%s] 处理流式请求", reqID)

	// 序列化请求
	jsonData, err := json.Marshal(hunyuanReq)
	if err != nil {
		logError("[reqID:%s] 序列化请求失败: %v", reqID, err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return err
	}

	// 创建请求
	httpReq, err := http.NewRequestWithContext(r.Context(), "POST", TargetURL, bytes.NewBuffer(jsonData))
	if err != nil {
		logError("[reqID:%s] 创建请求失败: %v", reqID, err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return err
	}

	// 设置请求头
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Model", hunyuanReq.Model)
	setCommonHeaders(httpReq)

	// 创建HTTP客户端
	client := getHTTPClient()
	
	// 发送请求
	resp, err := client.Do(httpReq)
	if err != nil {
		logError("[reqID:%s] 发送请求失败: %v", reqID, err)
		http.Error(w, "Failed to connect to API", http.StatusBadGateway)
		return err
	}
	defer resp.Body.Close()

	// 检查响应状态
	if resp.StatusCode != http.StatusOK {
		logError("[reqID:%s] API返回非200状态码: %d", reqID, resp.StatusCode)
		
		bodyBytes, _ := io.ReadAll(resp.Body)
		logError("[reqID:%s] 错误响应内容: %s", reqID, string(bodyBytes))
		
		http.Error(w, fmt.Sprintf("API error with status code: %d", resp.StatusCode), resp.StatusCode)
		return fmt.Errorf("API返回非200状态码: %d", resp.StatusCode)
	}

	// 设置响应头
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	// 创建响应ID和时间戳
	respID := fmt.Sprintf("chatcmpl-%s", getRandomString(10))
	createdTime := time.Now().Unix()
	
	// 创建读取器
	reader := bufio.NewReaderSize(resp.Body, 16384)
	
	// 创建Flusher
	flusher, ok := w.(http.Flusher)
	if !ok {
		logError("[reqID:%s] Streaming not supported", reqID)
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return fmt.Errorf("streaming not supported")
	}
	
	// 发送角色块
	roleChunk := createRoleChunk(respID, createdTime, hunyuanReq.Model)
	w.Write([]byte("data: " + string(roleChunk) + "\n\n"))
	flusher.Flush()
	
	// 持续读取响应
	for {
		// 添加超时检测
		select {
		case <-r.Context().Done():
			logWarn("[reqID:%s] 请求超时或被客户端取消", reqID)
			return fmt.Errorf("请求超时或被取消")
		default:
			// 继续处理
		}
		
		// 读取一行数据
		line, err := reader.ReadBytes('\n')
		if err != nil {
			if err != io.EOF {
				logError("[reqID:%s] 读取响应出错: %v", reqID, err)
				return err
			}
			break
		}
		
		// 处理数据行
		lineStr := string(line)
		if strings.HasPrefix(lineStr, "data: ") {
			jsonStr := strings.TrimPrefix(lineStr, "data: ")
			jsonStr = strings.TrimSpace(jsonStr)
			
			// 特殊处理[DONE]消息
			if jsonStr == "[DONE]" {
				logDebug("[reqID:%s] 收到[DONE]消息", reqID)
				w.Write([]byte("data: [DONE]\n\n"))
				flusher.Flush()
				break
			}
			
			// 解析混元响应
			var hunyuanResp HunyuanResponse
			if err := json.Unmarshal([]byte(jsonStr), &hunyuanResp); err != nil {
				logWarn("[reqID:%s] 解析JSON失败: %v, data: %s", reqID, err, jsonStr)
				continue
			}
			
			// 处理各种类型的内容
			for _, choice := range hunyuanResp.Choices {
				if choice.Delta.Content != "" {
					// 发送内容块
					contentChunk := createContentChunk(respID, createdTime, hunyuanReq.Model, choice.Delta.Content)
					w.Write([]byte("data: " + string(contentChunk) + "\n\n"))
					flusher.Flush()
				}
				
				if choice.Delta.ReasoningContent != "" {
					// 发送推理内容块
					reasoningChunk := createReasoningChunk(respID, createdTime, hunyuanReq.Model, choice.Delta.ReasoningContent)
					w.Write([]byte("data: " + string(reasoningChunk) + "\n\n"))
					flusher.Flush()
				}
				
				// 处理完成标志
				if choice.FinishReason != nil {
					finishReason := *choice.FinishReason
					if finishReason != "" {
						doneChunk := createDoneChunk(respID, createdTime, hunyuanReq.Model, finishReason)
						w.Write([]byte("data: " + string(doneChunk) + "\n\n"))
						flusher.Flush()
					}
				}
			}
		}
	}
	
	// 发送结束信号（如果没有正常结束）
	finishReason := "stop"
	doneChunk := createDoneChunk(respID, createdTime, hunyuanReq.Model, finishReason)
	w.Write([]byte("data: " + string(doneChunk) + "\n\n"))
	w.Write([]byte("data: [DONE]\n\n"))
	flusher.Flush()
	
	return nil
}

// 处理非流式请求
func handleNonStreamingRequest(w http.ResponseWriter, r *http.Request, hunyuanReq HunyuanRequest, reqID string) error {
	logInfo("[reqID:%s] 处理非流式请求", reqID)

	// 序列化请求
	jsonData, err := json.Marshal(hunyuanReq)
	if err != nil {
		logError("[reqID:%s] 序列化请求失败: %v", reqID, err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return err
	}

	// 创建请求
	httpReq, err := http.NewRequestWithContext(r.Context(), "POST", TargetURL, bytes.NewBuffer(jsonData))
	if err != nil {
		logError("[reqID:%s] 创建请求失败: %v", reqID, err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return err
	}

	// 设置请求头
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Model", hunyuanReq.Model)
	setCommonHeaders(httpReq)

	// 创建HTTP客户端
	client := getHTTPClient()
	
	// 发送请求
	resp, err := client.Do(httpReq)
	if err != nil {
		logError("[reqID:%s] 发送请求失败: %v", reqID, err)
		http.Error(w, "Failed to connect to API", http.StatusBadGateway)
		return err
	}
	defer resp.Body.Close()

	// 检查响应状态
	if resp.StatusCode != http.StatusOK {
		logError("[reqID:%s] API返回非200状态码: %d", reqID, resp.StatusCode)
		
		bodyBytes, _ := io.ReadAll(resp.Body)
		logError("[reqID:%s] 错误响应内容: %s", reqID, string(bodyBytes))
		
		http.Error(w, fmt.Sprintf("API error with status code: %d", resp.StatusCode), resp.StatusCode)
		return fmt.Errorf("API返回非200状态码: %d", resp.StatusCode)
	}

	// 读取完整的流式响应
	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		logError("[reqID:%s] 读取响应失败: %v", reqID, err)
		http.Error(w, "Failed to read API response", http.StatusInternalServerError)
		return err
	}

	// 解析流式响应并提取完整内容
	fullContent, reasoningContent, err := extractFullContentFromStream(bodyBytes, reqID)
	if err != nil {
		logError("[reqID:%s] 解析流式响应失败: %v", reqID, err)
		http.Error(w, "Failed to parse streaming response", http.StatusInternalServerError)
		return err
	}

	// 构建完整的非流式响应
	completionResponse := CompletionResponse{
		ID:      fmt.Sprintf("chatcmpl-%s", getRandomString(10)),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   hunyuanReq.Model,
		Choices: []struct {
			Index        int    `json:"index"`
			FinishReason string `json:"finish_reason"`
			Message      struct {
				Role             string `json:"role"`
				Content          string `json:"content"`
				ReasoningContent string `json:"reasoning_content,omitempty"`
			} `json:"message"`
		}{
			{
				Index:        0,
				FinishReason: "stop",
				Message: struct {
					Role             string `json:"role"`
					Content          string `json:"content"`
					ReasoningContent string `json:"reasoning_content,omitempty"`
				}{
					Role:             "assistant",
					Content:          fullContent,
					ReasoningContent: reasoningContent,
				},
			},
		},
		Usage: struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		}{
			PromptTokens:     len(formatMessages(hunyuanReq.Messages)) / 4,
			CompletionTokens: len(fullContent) / 4,
			TotalTokens:      (len(formatMessages(hunyuanReq.Messages)) + len(fullContent)) / 4,
		},
	}

	// 返回响应
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(completionResponse); err != nil {
		logError("[reqID:%s] 编码响应失败: %v", reqID, err)
		http.Error(w, "Failed to encode response", http.StatusInternalServerError)
		return err
	}

	return nil
}

// 从流式响应中提取完整内容
func extractFullContentFromStream(bodyBytes []byte, reqID string) (string, string, error) {
	bodyStr := string(bodyBytes)
	lines := strings.Split(bodyStr, "\n")
	
	// 内容累积器
	var contentBuilder strings.Builder
	var reasoningBuilder strings.Builder
	
	// 解析每一行
	for _, line := range lines {
		if strings.HasPrefix(line, "data: ") && !strings.Contains(line, "[DONE]") {
			jsonStr := strings.TrimPrefix(line, "data: ")
			jsonStr = strings.TrimSpace(jsonStr)
			
			// 解析JSON
			var hunyuanResp HunyuanResponse
			if err := json.Unmarshal([]byte(jsonStr), &hunyuanResp); err != nil {
				continue // 跳过无效JSON
			}
			
			// 提取内容和推理内容
			for _, choice := range hunyuanResp.Choices {
				if choice.Delta.Content != "" {
					contentBuilder.WriteString(choice.Delta.Content)
				}
				if choice.Delta.ReasoningContent != "" {
					reasoningBuilder.WriteString(choice.Delta.ReasoningContent)
				}
			}
		}
	}
	
	return contentBuilder.String(), reasoningBuilder.String(), nil
}

// 创建角色块
func createRoleChunk(id string, created int64, model string) []byte {
	chunk := StreamChunk{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   model,
		Choices: []struct {
			Index        int     `json:"index"`
			FinishReason *string `json:"finish_reason,omitempty"`
			Delta        struct {
				Role             string `json:"role,omitempty"`
				Content          string `json:"content,omitempty"`
				ReasoningContent string `json:"reasoning_content,omitempty"`
			} `json:"delta"`
		}{
			{
				Index: 0,
				Delta: struct {
					Role             string `json:"role,omitempty"`
					Content          string `json:"content,omitempty"`
					ReasoningContent string `json:"reasoning_content,omitempty"`
				}{
					Role: "assistant",
				},
			},
		},
	}

	data, _ := json.Marshal(chunk)
	return data
}

// 创建内容块
func createContentChunk(id string, created int64, model string, content string) []byte {
	chunk := StreamChunk{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   model,
		Choices: []struct {
			Index        int     `json:"index"`
			FinishReason *string `json:"finish_reason,omitempty"`
			Delta        struct {
				Role             string `json:"role,omitempty"`
				Content          string `json:"content,omitempty"`
				ReasoningContent string `json:"reasoning_content,omitempty"`
			} `json:"delta"`
		}{
			{
				Index: 0,
				Delta: struct {
					Role             string `json:"role,omitempty"`
					Content          string `json:"content,omitempty"`
					ReasoningContent string `json:"reasoning_content,omitempty"`
				}{
					Content: content,
				},
			},
		},
	}

	data, _ := json.Marshal(chunk)
	return data
}

// 创建推理内容块
func createReasoningChunk(id string, created int64, model string, reasoningContent string) []byte {
	chunk := StreamChunk{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   model,
		Choices: []struct {
			Index        int     `json:"index"`
			FinishReason *string `json:"finish_reason,omitempty"`
			Delta        struct {
				Role             string `json:"role,omitempty"`
				Content          string `json:"content,omitempty"`
				ReasoningContent string `json:"reasoning_content,omitempty"`
			} `json:"delta"`
		}{
			{
				Index: 0,
				Delta: struct {
					Role             string `json:"role,omitempty"`
					Content          string `json:"content,omitempty"`
					ReasoningContent string `json:"reasoning_content,omitempty"`
				}{
					ReasoningContent: reasoningContent,
				},
			},
		},
	}

	data, _ := json.Marshal(chunk)
	return data
}

// 创建完成块
func createDoneChunk(id string, created int64, model string, reason string) []byte {
	finishReason := reason
	chunk := StreamChunk{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   model,
		Choices: []struct {
			Index        int     `json:"index"`
			FinishReason *string `json:"finish_reason,omitempty"`
			Delta        struct {
				Role             string `json:"role,omitempty"`
				Content          string `json:"content,omitempty"`
				ReasoningContent string `json:"reasoning_content,omitempty"`
			} `json:"delta"`
		}{
			{
				Index:        0,
				FinishReason: &finishReason,
				Delta: struct {
					Role             string `json:"role,omitempty"`
					Content          string `json:"content,omitempty"`
					ReasoningContent string `json:"reasoning_content,omitempty"`
				}{},
			},
		},
	}

	data, _ := json.Marshal(chunk)
	return data
}

// 设置常见的请求头 - 参考Python版本
func setCommonHeaders(req *http.Request) {
	req.Header.Set("accept", "*/*")
	req.Header.Set("accept-language", "zh-CN,zh;q=0.9,en;q=0.8,zh-TW;q=0.7")
	req.Header.Set("authorization", "Bearer "+appConfig.BearerToken)
	req.Header.Set("dnt", "1")
	req.Header.Set("origin", "https://llm.hunyuan.tencent.com")
	req.Header.Set("polaris", "stream-server-online-sbs-10697")
	req.Header.Set("priority", "u=1, i")
	req.Header.Set("referer", "https://llm.hunyuan.tencent.com/")
	req.Header.Set("sec-ch-ua", "\"Chromium\";v=\"134\", \"Not:A-Brand\";v=\"24\", \"Google Chrome\";v=\"134\"")
	req.Header.Set("sec-ch-ua-mobile", "?0")
	req.Header.Set("sec-ch-ua-platform", "\"Windows\"")
	req.Header.Set("sec-fetch-dest", "empty")
	req.Header.Set("sec-fetch-mode", "cors")
	req.Header.Set("sec-fetch-site", "same-origin")
	req.Header.Set("staffname", "staryxzhang")
	req.Header.Set("wsid", "10697")
	req.Header.Set("user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36")
}

// 生成请求ID
func generateRequestID() string {
	return fmt.Sprintf("%x", time.Now().UnixNano())
}

// 生成随机字符串
func getRandomString(length int) string {
	const charset = "abcdefghijklmnopqrstuvwxyz0123456789"
	b := make([]byte, length)
	for i := range b {
		b[i] = charset[time.Now().UnixNano()%int64(len(charset))]
		time.Sleep(1 * time.Nanosecond)
	}
	return string(b)
}

// 格式化消息为字符串
func formatMessages(messages []APIMessage) string {
	var result strings.Builder
	for _, msg := range messages {
		result.WriteString(msg.Role)
		result.WriteString(": ")
		result.WriteString(contentToString(msg.Content))
		result.WriteString("\n")
	}
	return result.String()
}

// 获取两个整数中的最小值
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// 获取两个整数中的最大值
func max(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}