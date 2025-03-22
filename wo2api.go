package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"regexp"
	"runtime/debug"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

// 配置结构体用于存储命令行参数
type Config struct {
	Port            string // 代理服务器监听端口
	Address         string // 代理服务器监听地址
	LogLevel        string // 日志级别
	DevMode         bool   // 开发模式标志
	DiagnosticLevel string // 诊断级别：none, basic, full
	SaveResponses   bool   // 是否保存所有响应
	MaxRetries      int    // 最大重试次数
	Timeout         int    // 请求超时时间(秒)
}

// WoCloud API 目标URL，硬编码
const (
	TargetURL = "https://panservice.mail.wo.cn"
	ClientID  = "1001000035"
	Version   = "1.1.0" // 版本号
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
	flag.StringVar(&cfg.Port, "port", "5858", "Port to listen on")
	flag.StringVar(&cfg.Address, "address", "localhost", "Address to listen on")
	flag.StringVar(&cfg.LogLevel, "log-level", LogLevelInfo, "Log level (debug, info, warn, error)")
	flag.BoolVar(&cfg.DevMode, "dev", false, "Enable development mode with enhanced logging")
	flag.StringVar(&cfg.DiagnosticLevel, "diag", "none", "Diagnostic level: none, basic, full")
	flag.BoolVar(&cfg.SaveResponses, "save-responses", false, "Save all responses for analysis")
	flag.IntVar(&cfg.MaxRetries, "max-retries", 3, "Maximum number of retries for failed requests")
	flag.IntVar(&cfg.Timeout, "timeout", 300, "Request timeout in seconds")
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
	requestCounter      int64
	successCounter      int64
	errorCounter        int64
	parseErrorCounter   int64
	avgResponseTime     int64
	latencyHistogram    [10]int64 // 0-100ms, 100-200ms, ... >1s
	statusMetrics       sync.Map  // 记录不同状态码的计数
)

// 日志记录器
var (
	logger    *log.Logger
	logLevel  string
	logMutex  sync.Mutex
)

// 日志初始化
func initLogger(level string) {
	logger = log.New(os.Stdout, "[Wo2API] ", log.LstdFlags)
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

// WoCloud 历史记录格式
type WoCloudHistory struct {
	Query            string `json:"query"`
	RewriteQuery     string `json:"rewriteQuery"`
	UploadFileUrl    string `json:"uploadFileUrl"`
	Response         string `json:"response"`
	ReasoningContent string `json:"reasoningContent"`
	State            string `json:"state"`
	Key              string `json:"key"`
}

// WoCloud 请求格式
type WoCloudRequest struct {
	ModelId  int             `json:"modelId"`
	Input    string          `json:"input"`
	History  []WoCloudHistory `json:"history"`
}

// WoCloud 响应格式 - 增强版
type WoCloudResponse struct {
	Code             int         `json:"code"`
	Message          interface{} `json:"message"`
	Response         string      `json:"response"`
	ReasoningContent string      `json:"reasoningContent"`
	Finish           int         `json:"finish"`
	// 添加新字段增强兼容性
	Content          string      `json:"content,omitempty"`   // 兼容可能使用的另一种字段名
	Result           string      `json:"result,omitempty"`    // 兼容可能使用的另一种字段名
	Think            string      `json:"think,omitempty"`     // 兼容可能使用的思考内容字段
	Extra            map[string]interface{} `json:"-"`
}

// 自定义UnmarshalJSON方法，增强容错性
func (r *WoCloudResponse) UnmarshalJSON(data []byte) error {
	// 标准字段
	type StandardResponse WoCloudResponse
	
	// 临时结构，用于捕获所有字段
	var temp struct {
		StandardResponse
		Extra map[string]interface{} `json:"-"`
	}
	
	// 尝试标准解析
	if err := json.Unmarshal(data, &temp.StandardResponse); err != nil {
		// 尝试解析为map
		var rawMap map[string]interface{}
		if mapErr := json.Unmarshal(data, &rawMap); mapErr != nil {
			// 清除可能的BOM
			cleanData := bytes.TrimPrefix(data, []byte("\xef\xbb\xbf"))
			if cleanErr := json.Unmarshal(cleanData, &rawMap); cleanErr != nil {
				reqID := generateRequestID()
				logDebug("[reqID:%s] JSON解析失败: %v, 内容: %s", reqID, err, string(data[:min(len(data), 200)]))
				return err
			}
		}
		
		// 从map中提取关键字段
		for key, value := range rawMap {
			switch strings.ToLower(key) {
			case "code":
				switch v := value.(type) {
				case float64:
					temp.Code = int(v)
				case int:
					temp.Code = v
				case string:
					if c, err := strconv.Atoi(v); err == nil {
						temp.Code = c
					}
				}
			case "response", "content", "result":
				if str, ok := value.(string); ok && str != "" {
					temp.Response = str
				}
			case "reasoningcontent", "reasoning_content", "thinking", "think":
				if str, ok := value.(string); ok && str != "" {
					temp.ReasoningContent = str
				}
			case "finish", "done", "completed":
				switch v := value.(type) {
				case float64:
					temp.Finish = int(v)
				case int:
					temp.Finish = v
				case bool:
					if v {
						temp.Finish = 1
					}
				case string:
					if f, err := strconv.Atoi(v); err == nil {
						temp.Finish = f
					} else if v == "true" || v == "yes" {
						temp.Finish = 1
					}
				}
			}
		}
		
		temp.Extra = rawMap
	}
	
	// 复制回原结构
	*r = WoCloudResponse(temp.StandardResponse)
	r.Extra = temp.Extra
	
	// 优先级处理：如果 Response 为空但其他字段有值
	if r.Response == "" {
		if r.Content != "" {
			r.Response = r.Content
		} else if r.Result != "" {
			r.Response = r.Result
		}
	}
	
	// 如果 ReasoningContent 为空但 Think 有值
	if r.ReasoningContent == "" && r.Think != "" {
		r.ReasoningContent = r.Think
	}
	
	return nil
}

// DeepSeek 流式响应格式 - 修改以支持reasoning_content
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
			ReasoningContent string `json:"reasoning_content,omitempty"` // 使用reasoning_content而非think标签
		} `json:"delta"`
	} `json:"choices"`
}

// DeepSeek 非流式响应格式 - 修改以支持reasoning_content
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
			ReasoningContent string `json:"reasoning_content,omitempty"` // 使用reasoning_content
		} `json:"message"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

// WoCloud错误响应
type WoCloudError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// 请求计数和互斥锁，用于监控
var (
	requestCount uint64 = 0
	countMutex   sync.Mutex
)

// 启动指标报告器
func startMetricsReporter(interval time.Duration) {
	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		
		for {
			<-ticker.C
			
			reqCount := atomic.LoadInt64(&requestCounter)
			successCount := atomic.LoadInt64(&successCounter)
			errCount := atomic.LoadInt64(&errorCounter)
			parseErrCount := atomic.LoadInt64(&parseErrorCounter)
			
			// 仅当有请求时才输出指标
			if reqCount > 0 {
				avgTime := atomic.LoadInt64(&avgResponseTime)
				// 修复类型不匹配问题，确保使用相同类型计算成功率
				successRate := float64(0)
				if reqCount > 0 {
					successRate = float64(successCount) / float64(reqCount) * 100
				}
				
				logInfo("性能指标 - 请求总数: %d, 成功: %d (%.2f%%), 错误: %d, 解析错误: %d, 平均响应时间: %dms",
					reqCount, successCount, successRate, errCount, parseErrCount, avgTime/max(reqCount, 1))
				
				// 输出延迟直方图
				var latencyReport strings.Builder
				latencyReport.WriteString("延迟分布 - ")
				for i, count := range latencyHistogram {
					if count > 0 {
						if i < 9 {
							latencyReport.WriteString(fmt.Sprintf("%d-%dms: %d, ", i*100, (i+1)*100, count))
						} else {
							latencyReport.WriteString(fmt.Sprintf(">900ms: %d, ", count))
						}
					}
				}
				logInfo(strings.TrimSuffix(latencyReport.String(), ", "))
			}
		}
	}()
}

// 主入口函数
func main() {
	// 解析配置
	appConfig = parseFlags()
	
	// 初始化日志
	initLogger(appConfig.LogLevel)

	logInfo("启动服务: TargetURL=%s, Address=%s, Port=%s, Version=%s, LogLevel=%s",
		TargetURL, appConfig.Address, appConfig.Port, Version, appConfig.LogLevel)

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
		handleModelsRequest(w, r)
	})

	http.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		// 设置超时上下文
		ctx, cancel := context.WithTimeout(r.Context(), time.Duration(appConfig.Timeout)*time.Second)
		defer cancel()
		
		// 包含超时上下文的请求
		r = r.WithContext(ctx)
		
		// 添加恢复机制，防止panic
		defer func() {
			if r := recover(); r != nil {
				logError("处理请求时发生panic: %v\n%s", r, debug.Stack())
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
		countMutex.Lock()
		count := requestCount
		countMutex.Unlock()
		
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(fmt.Sprintf(`{"status":"ok","version":"%s","requests":%d}`, Version, count)))
	})
	
	// 添加版本端点
	http.HandleFunc("/version", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(fmt.Sprintf(`{"version":"%s"}`, Version)))
	})
	
	// 添加诊断端点
	http.HandleFunc("/diagnostics", func(w http.ResponseWriter, r *http.Request) {
		if !appConfig.DevMode {
			http.Error(w, "Diagnostics only available in development mode", http.StatusForbidden)
			return
		}
		
		reqCount := atomic.LoadInt64(&requestCounter)
		successCount := atomic.LoadInt64(&successCounter)
		errCount := atomic.LoadInt64(&errorCounter)
		parseErrCount := atomic.LoadInt64(&parseErrorCounter)
		
		// 计算成功率
		successRate := float64(0)
		if reqCount > 0 {
			successRate = float64(successCount) / float64(reqCount) * 100
		}
		
		diagnosticInfo := map[string]interface{}{
			"version":         Version,
			"start_time":      time.Now().Format(time.RFC3339),
			"requests":        reqCount,
			"success":         successCount,
			"errors":          errCount,
			"parse_errors":    parseErrCount,
			"success_rate":    fmt.Sprintf("%.2f%%", successRate),
			"avg_response_ms": atomic.LoadInt64(&avgResponseTime) / max(reqCount, 1),
		}
		
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(diagnosticInfo)
	})

	// 启动指标报告
	if appConfig.DevMode {
		startMetricsReporter(1 * time.Minute)
	}

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

// 从 OpenAI/DeepSeek 消息中提取用户消息和历史记录
func extractMessages(messages []APIMessage) (string, []WoCloudHistory) {
	reqID := generateRequestID()
	logDebug("[reqID:%s] 提取消息和历史记录", reqID)

	// 获取最后一条用户消息
	userMessage := ""
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			userMessage = contentToString(messages[i].Content)
			userMessage = strings.TrimSpace(userMessage)
			break
		}
	}

	// 构建历史记录
	var history []WoCloudHistory
	for i := 0; i < len(messages)-1; i++ {
		if messages[i].Role == "user" && i+1 < len(messages) && messages[i+1].Role == "assistant" {
			query := contentToString(messages[i].Content)
			response := contentToString(messages[i+1].Content)

			query = strings.TrimSpace(query)
			response = strings.TrimSpace(response)

			history = append(history, WoCloudHistory{
				Query:            query,
				RewriteQuery:     query,
				UploadFileUrl:    "",
				Response:         response,
				ReasoningContent: "", // 无法从标准消息中提取推理内容
				State:            "finish",
				Key:              fmt.Sprintf("%d", time.Now().UnixNano()),
			})
		}
	}

	logDebug("[reqID:%s] 提取的用户消息长度: %d", reqID, len(userMessage))
	logDebug("[reqID:%s] 提取的历史记录数量: %d", reqID, len(history))

	return userMessage, history
}

// 增强的WoCloud错误处理函数
func handleWoError(resp *http.Response) (*WoCloudError, error) {
    reqID := generateRequestID() // 为错误处理生成唯一ID
    logDebug("[reqID:%s] 处理WoCloud错误响应", reqID)
    
    contentType := resp.Header.Get("Content-Type")
    
    // 先读取整个响应体
    bodyBytes, err := io.ReadAll(resp.Body)
    if err != nil {
        return nil, fmt.Errorf("读取错误响应失败: %v", err)
    }
    
    bodyStr := string(bodyBytes)
    logDebug("[reqID:%s] 错误响应内容: %s", reqID, bodyStr)
    
    if strings.Contains(contentType, "text/event-stream") {
        // 流式响应中的错误
        lines := strings.Split(bodyStr, "\n")
        for _, line := range lines {
            if strings.HasPrefix(line, "data:") {
                jsonStr := strings.TrimPrefix(line, "data:")
                jsonStr = strings.TrimSpace(jsonStr)
                
                if jsonStr == "[DONE]" {
                    continue
                }
                
                var errorData WoCloudError
                if err := json.Unmarshal([]byte(jsonStr), &errorData); err != nil {
                    logDebug("[reqID:%s] 解析流式错误行失败: %v", reqID, err)
                    continue
                }
                
                if errorData.Code != "" && errorData.Code != "0" {
                    return &errorData, nil
                }
            }
        }
        
        // 如果没有找到具体错误信息，返回默认错误
        return &WoCloudError{
            Code:    fmt.Sprintf("HTTP_%d", resp.StatusCode),
            Message: fmt.Sprintf("Stream error with status code: %d", resp.StatusCode),
        }, nil
    } else {
        // 尝试解析为JSON错误
        var errorData WoCloudError
        if err := json.Unmarshal(bodyBytes, &errorData); err != nil {
            // 清理可能的BOM
            cleanBody := bytes.TrimPrefix(bodyBytes, []byte("\xef\xbb\xbf"))
            if err := json.Unmarshal(cleanBody, &errorData); err != nil {
                // 尝试用更宽松的方式解析
                var mapData map[string]interface{}
                if mapErr := json.Unmarshal(cleanBody, &mapData); mapErr == nil {
                    // 从map中提取错误信息
                    if code, ok := mapData["code"]; ok {
                        switch v := code.(type) {
                        case string:
                            errorData.Code = v
                        case float64:
                            errorData.Code = fmt.Sprintf("%d", int(v))
                        case int:
                            errorData.Code = fmt.Sprintf("%d", v)
                        }
                    }
                    
                    if message, ok := mapData["message"]; ok {
                        switch v := message.(type) {
                        case string:
                            errorData.Message = v
                        default:
                            errorData.Message = fmt.Sprintf("%v", v)
                        }
                    }
                    
                    return &errorData, nil
                }
                
                // 返回基于HTTP状态码的错误
                return &WoCloudError{
                    Code:    fmt.Sprintf("HTTP_%d", resp.StatusCode),
                    Message: fmt.Sprintf("Error with status code: %d and content: %s", resp.StatusCode, bodyStr),
                }, nil
            }
        }
        
        return &errorData, nil
    }
}

// 处理模型列表请求
func handleModelsRequest(w http.ResponseWriter, r *http.Request) {
	logInfo("处理模型列表请求")

	// 返回模型列表
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)

	modelsList := map[string]interface{}{
		"object": "list",
		"data": []map[string]interface{}{
			{
				"id":           "DeepSeek-R1",
				"object":       "model",
				"created":      time.Now().Unix(),
				"owned_by":     "ChinaUnicom",
				"capabilities": []string{"chat", "completions"},
			},
		},
	}

	json.NewEncoder(w).Encode(modelsList)
}

// 创建角色块 - 使用reasoning_content
func createRoleChunk(id string, created int64) []byte {
	chunk := StreamChunk{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   "DeepSeek-R1",
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

// 创建推理内容块 - 使用reasoning_content
func createReasoningChunk(id string, created int64, reasoningContent string) []byte {
	chunk := StreamChunk{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   "DeepSeek-R1",
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

// 创建内容块
func createContentChunk(id string, created int64, content string) []byte {
	chunk := StreamChunk{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   "DeepSeek-R1",
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

// 创建完成块
func createDoneChunk(id string, created int64, reason string) []byte {
	finishReason := reason
	chunk := StreamChunk{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   "DeepSeek-R1",
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

// 创建错误块
func createErrorChunk(id string, created int64, errorMsg string) []byte {
	chunk := map[string]interface{}{
		"error": map[string]interface{}{
			"message": errorMsg,
			"type":    "api_error",
		},
	}

	data, _ := json.Marshal(chunk)
	return data
}

// 处理聊天补全请求
func handleChatCompletionRequest(w http.ResponseWriter, r *http.Request) {
	reqID := generateRequestID()
	startTime := time.Now()
	logInfo("[reqID:%s] 处理聊天补全请求", reqID)

	// 从请求头中提取令牌
	token, err := extractToken(r)
	if err != nil {
		logError("[reqID:%s] 提取令牌失败: %v", reqID, err)
		http.Error(w, fmt.Sprintf("Authorization error: %v", err), http.StatusUnauthorized)
		return
	}

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

	// 获取最后一条用户消息和历史记录
	userMessage, history := extractMessages(apiReq.Messages)
	if userMessage == "" {
		logError("[reqID:%s] 未找到有效的用户消息", reqID)
		http.Error(w, "No valid user message found", http.StatusBadRequest)
		return
	}

	// 转发请求到 WoCloud API
	var responseErr error
	if apiReq.Stream {
		responseErr = handleStreamingRequest(w, r, userMessage, history, token, reqID)
	} else {
		responseErr = handleNonStreamingRequest(w, r, userMessage, history, token, reqID)
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

// 重试机制的HTTP请求函数
func doRequestWithRetry(req *http.Request, client *http.Client, maxRetries int) (*http.Response, error) {
    var resp *http.Response
    var err error
    reqID := generateRequestID()
    
    for i := 0; i < maxRetries; i++ {
        resp, err = client.Do(req)
        if err == nil && resp.StatusCode == http.StatusOK {
            logDebug("[reqID:%s] HTTP请求成功", reqID)
            return resp, nil
        }
        
        if resp != nil {
            resp.Body.Close()
        }
        
        logWarn("[reqID:%s] HTTP请求失败（尝试%d/%d）: %v", reqID, i+1, maxRetries, err)
        
        // 避免最后一次失败后等待
        if i < maxRetries-1 {
            // 指数退避
            backoffTime := time.Duration(100*(1<<i)) * time.Millisecond
            logDebug("[reqID:%s] 等待 %v 后重试", reqID, backoffTime)
            time.Sleep(backoffTime)
        }
    }
    
    // 返回适当的错误信息
    if err != nil {
        return nil, fmt.Errorf("在%d次尝试后HTTP请求仍然失败: %v", maxRetries, err)
    }
    
    return nil, fmt.Errorf("在%d次尝试后HTTP请求返回非200状态码: %d", maxRetries, resp.StatusCode)
}

// 清理JSON字符串，去除可能导致JSON编码失败的字符
func sanitizeJsonString(input string, reqID string) string {
	if input == "" {
		return input
	}
	
	// 记录原始长度
	originalLen := len(input)
	
	// 移除控制字符（除了常见的换行、回车、制表符）
	cleanStr := strings.Map(func(r rune) rune {
		if r < 32 && r != '\n' && r != '\r' && r != '\t' {
			return -1 // 删除字符
		}
		return r
	}, input)
	
	// 处理不成对的引号和转义字符
	var result strings.Builder
	inBackslash := false
	openQuote := false
	
	for _, ch := range cleanStr {
		switch {
		case inBackslash:
			inBackslash = false
			result.WriteRune(ch)
		case ch == '\\':
			inBackslash = true
			result.WriteRune(ch)
		case ch == '"':
			openQuote = !openQuote
			result.WriteRune(ch)
		default:
			result.WriteRune(ch)
		}
	}
	
	// 确保JSON合法性
	cleanedStr := result.String()
	
	// 在关键位置添加额外日志，帮助排查问题
	if originalLen != len(cleanedStr) {
		logDebug("[reqID:%s] JSON清理: 原始长度=%d, 清理后长度=%d", 
			reqID, originalLen, len(cleanedStr))
		
		// 输出清理前后的部分样本用于比较
		sampleSize := 50
		if originalLen < sampleSize {
			sampleSize = originalLen
		}
		
		logDebug("[reqID:%s] 清理前样本: %s", reqID, input[:sampleSize])
		if len(cleanedStr) < sampleSize {
			sampleSize = len(cleanedStr)
		}
		
		if sampleSize > 0 {
			logDebug("[reqID:%s] 清理后样本: %s", reqID, cleanedStr[:sampleSize])
		}
	}
	
	// 返回清理后的字符串
	return cleanedStr
}

// 截断字符串到指定长度
func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

// 从JSON字符串中递归提取所有字符串值
func extractAllStringValues(obj map[string]interface{}, values map[string]string, prefix string) {
	for k, v := range obj {
		path := prefix
		if prefix != "" {
			path += "." + k
		} else {
			path = k
		}
		
		switch val := v.(type) {
		case string:
			values[path] = val
		case map[string]interface{}:
			extractAllStringValues(val, values, path)
		case []interface{}:
			for i, item := range val {
				if mapItem, ok := item.(map[string]interface{}); ok {
					extractAllStringValues(mapItem, values, fmt.Sprintf("%s[%d]", path, i))
				} else if strItem, ok := item.(string); ok {
					values[fmt.Sprintf("%s[%d]", path, i)] = strItem
				}
			}
		}
	}
}

// 从流式响应中查找最有可能的完整内容
func findMostLikelyContent(lines []string, reqID string) string {
	// 如果行数少于2，无法使用此方法
	if len(lines) < 2 {
		return ""
	}
	
	// 先解析所有有效的JSON行
	var jsonObjects []map[string]interface{}
	
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		
		jsonStr := strings.TrimPrefix(line, "data:")
		jsonStr = strings.TrimSpace(jsonStr)
		
		if jsonStr == "[DONE]" || jsonStr == "" {
			continue
		}
		
		var obj map[string]interface{}
		if err := json.Unmarshal([]byte(jsonStr), &obj); err != nil {
			// 尝试修复JSON
			fixedJson := sanitizeJsonString(jsonStr, reqID)
			if err := json.Unmarshal([]byte(fixedJson), &obj); err != nil {
				continue
			}
		}
		
		jsonObjects = append(jsonObjects, obj)
	}
	
	// 如果没有有效JSON对象，返回空
	if len(jsonObjects) == 0 {
		return ""
	}
	
	// 提取每个对象中的所有字符串值
	allValues := make([]map[string]string, len(jsonObjects))
	
	for i, obj := range jsonObjects {
		allValues[i] = make(map[string]string)
		extractAllStringValues(obj, allValues[i], "")
	}
	
	// 分析哪些路径出现频率最高
	pathFrequency := make(map[string]int)
	for _, values := range allValues {
		for path := range values {
			pathFrequency[path]++
		}
	}
	
	// 找出最常见的路径
	var mostCommonPath string
	var maxFrequency int = 0
	
	for path, freq := range pathFrequency {
		if freq > maxFrequency {
			mostCommonPath = path
			maxFrequency = freq
		}
	}
	
	// 找到所有最常见路径的值，并选择最长的一个
	var longestValue string
	var maxLength int = 0
	
	for _, values := range allValues {
		if value, ok := values[mostCommonPath]; ok {
			if len(value) > maxLength {
				longestValue = value
				maxLength = len(value)
			}
		}
	}
	
	// 记录选择结果
	logInfo("[reqID:%s] 找到最可能的内容路径: %s, 出现频率: %d/%d, 最大长度: %d",
		reqID, mostCommonPath, maxFrequency, len(jsonObjects), maxLength)
	
	return longestValue
}

// 尝试从JSON字符串中提取响应内容
func extractResponseContent(jsonStr string, reqID string) string {
	// 尝试几种正则表达式模式
	patterns := []string{
		`"response"\s*:\s*"([^"]*)"`,
		`"response":"(.*?)"`,
		`response":"([^"]+)"`,
	}
	
	for _, pattern := range patterns {
		re := regexp.MustCompile(pattern)
		matches := re.FindStringSubmatch(jsonStr)
		if len(matches) > 1 && matches[1] != "" {
			return matches[1]
		}
	}
	
	return ""
}

// 根据API响应的特征智能提取内容
func intelligentContentExtraction(body []byte, reqID string) string {
	bodyStr := string(body)
	lines := strings.Split(bodyStr, "\n")
	
	// 方法1: 使用找到最常见路径的方法
	content := findMostLikelyContent(lines, reqID)
	if content != "" && len(content) > 10 {
		logInfo("[reqID:%s] 使用最可能内容路径方法提取到内容，长度=%d", reqID, len(content))
		return content
	}
	
	// 方法2: 使用最长响应行策略
	var longestResponse string
	var maxLength int = 0
	
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		
		jsonStr := strings.TrimPrefix(line, "data:")
		jsonStr = strings.TrimSpace(jsonStr)
		
		if jsonStr == "[DONE]" || jsonStr == "" {
			continue
		}
		
		var obj map[string]interface{}
		if err := json.Unmarshal([]byte(jsonStr), &obj); err != nil {
			continue
		}
		
		// 提取response字段
		if resp, ok := obj["response"]; ok {
			if respStr, ok := resp.(string); ok {
				if len(respStr) > maxLength {
					longestResponse = respStr
					maxLength = len(respStr)
				}
			}
		}
	}
	
	if longestResponse != "" {
		logInfo("[reqID:%s] 使用最长响应策略提取到内容，长度=%d", reqID, len(longestResponse))
		return longestResponse
	}
	
	// 方法3: 使用正则表达式
	patterns := []string{
		`"response"\s*:\s*"((?:.|\n)*?)"`,
		`"response":"([^"]*)"`,
		`response":"([^"]+)"`,
	}
	
	for _, pattern := range patterns {
		re := regexp.MustCompile(pattern)
		matches := re.FindAllStringSubmatch(bodyStr, -1)
		
		if len(matches) > 0 {
			var longestMatch string
			var maxMatchLength int = 0
			
			for _, match := range matches {
				if len(match) > 1 && len(match[1]) > maxMatchLength {
					// 处理转义
					content := match[1]
					content = strings.Replace(content, "\\n", "\n", -1)
					content = strings.Replace(content, "\\\"", "\"", -1)
					content = strings.Replace(content, "\\\\", "\\", -1)
					
					longestMatch = content
					maxMatchLength = len(content)
				}
			}
			
			if longestMatch != "" {
				logInfo("[reqID:%s] 使用正则表达式方法提取到内容，长度=%d", reqID, len(longestMatch))
				return longestMatch
			}
		}
	}
	
	// 未能提取到有效内容
	return ""
}

// 将原始响应保存为分析文件
func saveResponseForAnalysis(body []byte, reqID string) string {
	diagDir := "diagnostics"
	os.MkdirAll(diagDir, 0755)
	
	// 保存原始响应
	rawPath := filepath.Join(diagDir, fmt.Sprintf("raw_response_%s.txt", reqID))
	if err := os.WriteFile(rawPath, body, 0644); err != nil {
		logError("[reqID:%s] 保存原始响应文件失败: %v", reqID, err)
		return ""
	}
	
	// 创建分析文件
	analysisPath := filepath.Join(diagDir, fmt.Sprintf("analysis_%s.txt", reqID))
	f, err := os.Create(analysisPath)
	if err != nil {
		logError("[reqID:%s] 创建分析文件失败: %v", reqID, err)
		return rawPath // 至少返回原始文件路径
	}
	defer f.Close()
	
	// 写入分析头部
	fmt.Fprintf(f, "WoCloud 响应分析 - 请求ID: %s\n", reqID)
	fmt.Fprintf(f, "分析时间: %s\n", time.Now().Format(time.RFC3339))
	fmt.Fprintf(f, "原始响应大小: %d 字节\n\n", len(body))
	
	// 解析响应行
	bodyStr := string(body)
	lines := strings.Split(bodyStr, "\n")
	
	// 统计信息
	var stats struct {
		TotalLines        int
		DataLines         int
		ValidJSON         int
		InvalidJSON       int
		ResponseLines     int
		MaxResponseLength int
	}
	
	stats.TotalLines = len(lines)
	
	// 记录所有响应内容的长度
	responseLengths := make([]int, 0)
	
	// 分析每一行
	for i, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		
		if !strings.HasPrefix(line, "data:") {
			fmt.Fprintf(f, "行 %d: 非data前缀行: %s\n", i+1, truncateString(line, 100))
			continue
		}
		
		stats.DataLines++
		jsonStr := strings.TrimPrefix(line, "data:")
		jsonStr = strings.TrimSpace(jsonStr)
		
		if jsonStr == "[DONE]" {
			fmt.Fprintf(f, "行 %d: [DONE]标记\n", i+1)
			continue
		}
		
		if jsonStr == "" {
			fmt.Fprintf(f, "行 %d: 空数据行\n", i+1)
			continue
		}
		
		// 尝试解析JSON
		var obj map[string]interface{}
		if err := json.Unmarshal([]byte(jsonStr), &obj); err != nil {
			stats.InvalidJSON++
			fmt.Fprintf(f, "行 %d: 无效JSON: %s\n  错误: %v\n", i+1, truncateString(jsonStr, 100), err)
			continue
		}
		
		stats.ValidJSON++
		
		// 检查是否有response字段
		if resp, ok := obj["response"]; ok {
			if respStr, ok := resp.(string); ok {
				stats.ResponseLines++
				respLen := len(respStr)
				responseLengths = append(responseLengths, respLen)
				
				if respLen > stats.MaxResponseLength {
					stats.MaxResponseLength = respLen
				}
				
				// 记录响应内容摘要
				fmt.Fprintf(f, "行 %d: 包含response字段，长度=%d, 内容: %s\n", 
					i+1, respLen, truncateString(respStr, 100))
			} else {
				fmt.Fprintf(f, "行 %d: response字段非字符串类型: %T\n", i+1, resp)
			}
		} else {
			// 打印完整JSON对象
			prettyJSON, _ := json.MarshalIndent(obj, "", "  ")
			maxJSON := truncateString(string(prettyJSON), 500)
			fmt.Fprintf(f, "行 %d: 不包含response字段，完整JSON: \n%s\n", i+1, maxJSON)
		}
	}
	
	// 写入统计信息
	fmt.Fprintf(f, "\n统计信息:\n")
	fmt.Fprintf(f, "总行数: %d\n", stats.TotalLines)
	fmt.Fprintf(f, "data:前缀行数: %d\n", stats.DataLines)
	fmt.Fprintf(f, "有效JSON行数: %d\n", stats.ValidJSON)
	fmt.Fprintf(f, "无效JSON行数: %d\n", stats.InvalidJSON)
	fmt.Fprintf(f, "包含response字段的行数: %d\n", stats.ResponseLines)
	fmt.Fprintf(f, "最大response长度: %d\n", stats.MaxResponseLength)
	
	// 响应长度分布
	if len(responseLengths) > 0 {
		fmt.Fprintf(f, "\nresponse长度分布:\n")
		
		// 排序长度
		sort.Ints(responseLengths)
		
		// 计算分位数
		p25 := responseLengths[len(responseLengths)*1/4]
		p50 := responseLengths[len(responseLengths)*2/4]
		p75 := responseLengths[len(responseLengths)*3/4]
		p100 := responseLengths[len(responseLengths)-1]
		
		fmt.Fprintf(f, "最小值: %d\n", responseLengths[0])
		fmt.Fprintf(f, "25%%分位数: %d\n", p25)
		fmt.Fprintf(f, "中位数: %d\n", p50)
		fmt.Fprintf(f, "75%%分位数: %d\n", p75)
		fmt.Fprintf(f, "最大值: %d\n", p100)
		
		// 记录所有响应长度
		fmt.Fprintf(f, "\n所有response长度: %v\n", responseLengths)
	}
	
	// 尝试使用各种方法提取内容
	fmt.Fprintf(f, "\n尝试使用各种方法提取内容:\n")
	
	// 方法1: 最常见路径
	content1 := findMostLikelyContent(lines, reqID)
	if content1 != "" {
		fmt.Fprintf(f, "1. 使用最常见路径方法提取到内容，长度=%d\n", len(content1))
		fmt.Fprintf(f, "   前100个字符: %s\n", truncateString(content1, 100))
	} else {
		fmt.Fprintf(f, "1. 最常见路径方法未能提取到内容\n")
	}
	
	// 方法2: 正则表达式
	patterns := []string{
		`"response"\s*:\s*"((?:.|\n)*?)"`,
		`"response":"([^"]*)"`,
		`response":"([^"]+)"`,
	}
	
	for i, pattern := range patterns {
		re := regexp.MustCompile(pattern)
		matches := re.FindAllStringSubmatch(bodyStr, -1)
		
		if len(matches) > 0 {
			var longestMatch string
			var maxMatchLength int = 0
			
			for _, match := range matches {
				if len(match) > 1 && len(match[1]) > maxMatchLength {
					longestMatch = match[1]
					maxMatchLength = len(match[1])
				}
			}
			
			if longestMatch != "" {
				fmt.Fprintf(f, "2.%d 使用正则表达式 '%s' 提取到内容，长度=%d\n", 
					i+1, truncateString(pattern, 30), len(longestMatch))
				fmt.Fprintf(f, "    前100个字符: %s\n", truncateString(longestMatch, 100))
			}
		} else {
			fmt.Fprintf(f, "2.%d 正则表达式 '%s' 未找到匹配\n", i+1, truncateString(pattern, 30))
		}
	}
	
	// 方法3: 使用最长响应行策略
	var longestResponse string
	var maxLength int = 0
	
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		
		jsonStr := strings.TrimPrefix(line, "data:")
		jsonStr = strings.TrimSpace(jsonStr)
		
		if jsonStr == "[DONE]" || jsonStr == "" {
			continue
		}
		
		var obj map[string]interface{}
		if err := json.Unmarshal([]byte(jsonStr), &obj); err != nil {
			continue
		}
		
		// 提取response字段
		if resp, ok := obj["response"]; ok {
			if respStr, ok := resp.(string); ok {
				if len(respStr) > maxLength {
					longestResponse = respStr
					maxLength = len(respStr)
				}
			}
		}
	}
	
	if longestResponse != "" {
		fmt.Fprintf(f, "3. 使用最长响应策略提取到内容，长度=%d\n", len(longestResponse))
		fmt.Fprintf(f, "   前100个字符: %s\n", truncateString(longestResponse, 100))
	} else {
		fmt.Fprintf(f, "3. 最长响应策略未能提取到内容\n")
	}
	
	logInfo("[reqID:%s] 完整诊断分析已保存到: %s", reqID, analysisPath)
	return analysisPath
}

// 增强版诊断流式响应
func enhancedDiagnoseChatResponse(body []byte, reqID string) {
	// 如果没有启用诊断，则直接返回
	if appConfig.DiagnosticLevel == "none" && !appConfig.DevMode {
		return
	}
	
	// 保存完整响应内容和进行详细分析 - 不使用返回值
	saveResponseForAnalysis(body, reqID)
	
	// 如果在开发模式下，进行更深入分析
	if appConfig.DevMode {
		// 提取并保存最可能的完整内容
		content := intelligentContentExtraction(body, reqID)
		if content != "" {
			contentPath := filepath.Join("diagnostics", fmt.Sprintf("extracted_content_%s.txt", reqID))
			os.WriteFile(contentPath, []byte(content), 0644)
			logInfo("[reqID:%s] 保存提取的完整内容到: %s", reqID, contentPath)
		}
		
		// 打印流式响应的结构信息
		analyzeStreamStructure(body, reqID)
	}
}

// 分析流式响应结构
func analyzeStreamStructure(body []byte, reqID string) {
	bodyStr := string(body)
	lines := strings.Split(bodyStr, "\n")
	
	// 统计各种类型的行数
	dataLines := 0
	jsonLines := 0
	doneLines := 0
	
	// 响应长度增长模式
	responseLengths := make([]int, 0)
	
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		
		dataLines++
		jsonStr := strings.TrimPrefix(line, "data:")
		jsonStr = strings.TrimSpace(jsonStr)
		
		if jsonStr == "[DONE]" {
			doneLines++
			continue
		}
		
		if jsonStr == "" {
			continue
		}
		
		var obj map[string]interface{}
		if err := json.Unmarshal([]byte(jsonStr), &obj); err != nil {
			continue
		}
		
		jsonLines++
		
		// 记录响应长度
		if resp, ok := obj["response"]; ok {
			if respStr, ok := resp.(string); ok {
				responseLengths = append(responseLengths, len(respStr))
			}
		}
	}
	
	// 分析响应长度增长模式
	isIncremental := true
	for i := 1; i < len(responseLengths); i++ {
		if responseLengths[i] < responseLengths[i-1] {
			isIncremental = false
			break
		}
	}
	
	// 打印分析结果
	logInfo("[reqID:%s] 流式响应结构分析: 总行数=%d, data行数=%d, JSON行数=%d, DONE行数=%d", 
		reqID, len(lines), dataLines, jsonLines, doneLines)
	
	if len(responseLengths) > 0 {
		logInfo("[reqID:%s] 响应长度模式: %v, 增量模式=%v", reqID, responseLengths[:min(10, len(responseLengths))], isIncremental)
	}
}

// 将流式响应解析为非流式响应 - 完全重写版
func parseStreamResponseAsNonStream(body []byte, reqID string) (*WoCloudResponse, error) {
	logInfo("[reqID:%s] 解析流式响应为非流式", reqID)
	parseStartTime := time.Now()
	
	// 如果配置了诊断，保存响应内容
	if appConfig.DiagnosticLevel != "none" || appConfig.SaveResponses {
		enhancedDiagnoseChatResponse(body, reqID)
	}
	
	bodyStr := string(body)
	lines := strings.Split(bodyStr, "\n")
	
	// 统计变量
	totalLines := 0
	validJsonLines := 0
	
	// 初始化完整响应对象
	fullResponse := &WoCloudResponse{
		Code:             0,
		Response:         "",
		ReasoningContent: "", 
		Finish:           1, // 默认设置为完成状态
	}
	
	// 创建一个有序map存储响应行，确保按顺序处理
	responseMap := make(map[int]string)
	var maxIndex int = 0
	
	// 第一步：收集所有的data:行并解析
	for i, line := range lines {
		line = strings.TrimSpace(line)
		
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		
		totalLines++
		jsonStr := strings.TrimPrefix(line, "data:")
		jsonStr = strings.TrimSpace(jsonStr)
		
		if jsonStr == "[DONE]" {
			continue
		}
		
		// 解析JSON
		var respObj map[string]interface{}
		if err := json.Unmarshal([]byte(jsonStr), &respObj); err != nil {
			// 尝试修复JSON格式
			fixedJson := sanitizeJsonString(jsonStr, reqID)
			if err := json.Unmarshal([]byte(fixedJson), &respObj); err != nil {
				// 跳过无效的JSON
				continue
			}
		}
		
		validJsonLines++
		
		// 提取响应内容
		if resp, ok := respObj["response"]; ok {
			if respStr, ok := resp.(string); ok && respStr != "" {
				// 存储响应，使用行号作为键确保顺序
				responseMap[i] = respStr
				if i > maxIndex {
					maxIndex = i
				}
			}
		}
	}
	
	// 第二步：如果没有找到任何有效响应，尝试使用正则表达式
	if len(responseMap) == 0 {
		logWarn("[reqID:%s] 没有找到有效的响应行，尝试使用正则表达式", reqID)
		
		// 尝试多种正则表达式模式
		patterns := []string{
			`"response"\s*:\s*"((?:.|\n)*?)"`,
			`"response":"([^"]*)"`,
			`response":"([^"]+)"`,
		}
		
		for _, pattern := range patterns {
			re := regexp.MustCompile(pattern)
			matches := re.FindAllStringSubmatch(bodyStr, -1)
			
			if len(matches) > 0 {
				// 收集所有匹配
				for i, match := range matches {
					if len(match) > 1 && match[1] != "" {
						// 处理转义
						content := match[1]
						content = strings.Replace(content, "\\n", "\n", -1)
						content = strings.Replace(content, "\\\"", "\"", -1)
						content = strings.Replace(content, "\\\\", "\\", -1)
						
						responseMap[i] = content
						if i > maxIndex {
							maxIndex = i
						}
					}
				}
				
				if len(responseMap) > 0 {
					break // 找到匹配就停止
				}
			}
		}
	}
	
	// 第三步：确定合并策略
	if len(responseMap) == 0 {
		// 没有找到任何响应内容，尝试使用智能提取
		fullResponse.Response = intelligentContentExtraction(body, reqID)
		if fullResponse.Response == "" {
			// 智能提取也失败，使用默认消息
			fullResponse.Response = "抱歉，无法获取有效回复。请稍后再试。"
			logWarn("[reqID:%s] 无法提取响应内容，使用默认消息", reqID)
			atomic.AddInt64(&parseErrorCounter, 1)
		} else {
			logInfo("[reqID:%s] 使用智能提取获取响应内容，长度=%d", reqID, len(fullResponse.Response))
		}
	} else {
		// 尝试确定响应模式
		
		// 模式1: 增量流式响应（每行都是前面行的超集）
		// 模式2: 完整消息流式响应（每行都是完整消息）
		// 默认假设是模式1
		isIncrementalMode := true
		
		// 检查响应模式
		if len(responseMap) >= 2 {
			// 获取有序的键
			var keys []int
			for k := range responseMap {
				keys = append(keys, k)
			}
			sort.Ints(keys)
			
			// 检查是否是增量模式
			for i := 1; i < len(keys); i++ {
				prev := responseMap[keys[i-1]]
				curr := responseMap[keys[i]]
				
				// 如果当前行不是前一行的超集，可能不是增量模式
				if !strings.HasPrefix(curr, prev) && len(curr) >= len(prev) {
					isIncrementalMode = false
					break
				}
			}
		}
		
		// 根据不同模式合并响应
		if isIncrementalMode {
			// 增量模式：使用最后一行（最完整的）
			fullResponse.Response = responseMap[maxIndex]
			logInfo("[reqID:%s] 使用增量模式，选择最后一行作为完整响应，长度=%d", reqID, len(fullResponse.Response))
		} else {
			// 非增量模式，尝试两种策略：
			// 1. 使用最长的响应
			// 2. 合并所有响应（去重）
			
			// 首先尝试找最长响应
			var longestResponse string
			var longestLength int = 0
			
			for _, resp := range responseMap {
				if len(resp) > longestLength {
					longestResponse = resp
					longestLength = len(resp)
				}
			}
			
			// 如果最长响应足够长（超过50个字符），直接使用
			if longestLength > 50 {
				fullResponse.Response = longestResponse
				logInfo("[reqID:%s] 使用最长响应，长度=%d", reqID, longestLength)
			} else {
				// 最长响应不够长，尝试合并
				// 先获取有序的响应
				var keys []int
				for k := range responseMap {
					keys = append(keys, k)
				}
				sort.Ints(keys)
				
				var combinedResponse strings.Builder
				var usedSubstrings = make(map[string]bool)
				
				for _, k := range keys {
					resp := responseMap[k]
					// 只添加新内容
					if !usedSubstrings[resp] {
						combinedResponse.WriteString(resp)
						usedSubstrings[resp] = true
					}
				}
				
				combinedContent := combinedResponse.String()
				
				if len(combinedContent) > longestLength {
					fullResponse.Response = combinedContent
					logInfo("[reqID:%s] 使用合并响应，长度=%d", reqID, len(combinedContent))
				} else {
					fullResponse.Response = longestResponse
					logInfo("[reqID:%s] 合并后仍不如最长响应，使用最长响应，长度=%d", reqID, longestLength)
				}
			}
		}
		
		// 检查最终响应是否太短
		if len(fullResponse.Response) < 10 {
			logWarn("[reqID:%s] 最终响应内容过短(%d字符)，可能不完整", reqID, len(fullResponse.Response))
			
			// 尝试智能提取
			smartExtraction := intelligentContentExtraction(body, reqID)
			if smartExtraction != "" && len(smartExtraction) > len(fullResponse.Response) {
				fullResponse.Response = smartExtraction
				logInfo("[reqID:%s] 使用智能提取替换过短的响应，新长度=%d", reqID, len(smartExtraction))
			}
			
			// 保存响应数据用于后续分析
			if appConfig.DevMode {
				debugDir := "debug"
				os.MkdirAll(debugDir, 0755)
				
				debugPath := filepath.Join(debugDir, fmt.Sprintf("short_response_%s.json", reqID))
				debug, _ := json.MarshalIndent(responseMap, "", "  ")
				os.WriteFile(debugPath, debug, 0644)
				
				logInfo("[reqID:%s] 已保存短响应调试信息到: %s", reqID, debugPath)
			}
		}
	}
	
	parseElapsed := time.Since(parseStartTime).Milliseconds()
	logInfo("[reqID:%s] 流式解析完成: 总行数=%d, 有效JSON行数=%d, 最终响应长度=%d, 解析耗时: %dms", 
		reqID, totalLines, validJsonLines, len(fullResponse.Response), parseElapsed)
		
	return fullResponse, nil
}

// 流式请求 - 支持reasoning_content传递
func handleStreamingRequest(w http.ResponseWriter, r *http.Request, userMessage string, history []WoCloudHistory, token string, reqID string) error {
	logInfo("[reqID:%s] 处理流式请求", reqID)

	// 构建 WoCloud 请求
	woReq := WoCloudRequest{
		ModelId: 1,
		Input:   userMessage,
		History: history,
	}

	jsonData, err := json.Marshal(woReq)
	if err != nil {
		logError("[reqID:%s] 序列化请求失败: %v", reqID, err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return err
	}

	// 创建请求
	woURL := TargetURL + "/wohome/ai/assistant/query"
	httpReq, err := http.NewRequest("POST", woURL, bytes.NewBuffer(jsonData))
	if err != nil {
		logError("[reqID:%s] 创建请求失败: %v", reqID, err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return err
	}

	// 设置请求头
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")
	httpReq.Header.Set("Origin", TargetURL)
	httpReq.Header.Set("Referer", TargetURL+"/h5/wocloud_ai/?modelType=1")
	httpReq.Header.Set("User-Agent", "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1")
	httpReq.Header.Set("X-YP-Access-Token", token)
	httpReq.Header.Set("X-YP-Client-ID", ClientID)

	// 为每个请求创建一个新的HTTP客户端，避免共享连接池导致的干扰
	client := &http.Client{
		Timeout: time.Duration(appConfig.Timeout) * time.Second,
		Transport: &http.Transport{
			MaxIdleConnsPerHost: 100,
			IdleConnTimeout:     90 * time.Second,
		},
	}
	
	// 记录请求细节
	logDebug("[reqID:%s] 流式请求详情: URL=%s, 用户消息长度=%d, 历史记录数=%d", 
		reqID, woURL, len(userMessage), len(history))
	
	// 使用重试机制
	resp, err := doRequestWithRetry(httpReq, client, appConfig.MaxRetries)
	if err != nil {
		logError("[reqID:%s] 发送请求失败: %v", reqID, err)
		http.Error(w, "Failed to connect to API", http.StatusBadGateway)
		return err
	}
	defer resp.Body.Close()

	// 检查响应状态
	if resp.StatusCode != http.StatusOK {
		logError("[reqID:%s] API返回非200状态码: %d", reqID, resp.StatusCode)
		
		errorInfo, err := handleWoError(resp)
		if err != nil {
			logError("[reqID:%s] 解析错误响应失败: %v", reqID, err)
			http.Error(w, "Failed to parse error response", http.StatusBadGateway)
			return err
		}
		
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(resp.StatusCode)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"error": errorInfo,
		})
		return fmt.Errorf("API返回错误: %s", errorInfo.Message)
	}

	// 设置响应头
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	// 生成响应 ID
	respID := fmt.Sprintf("chatcmpl-%s", generateRandomID())
	createdTime := time.Now().Unix()

	// 发送角色信息
	roleChunk := createRoleChunk(respID, createdTime)
	w.Write([]byte("data: " + string(roleChunk) + "\n\n"))
	flusher, ok := w.(http.Flusher)
	if !ok {
		logError("[reqID:%s] Streaming not supported", reqID)
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return fmt.Errorf("streaming not supported")
	}
	flusher.Flush()

	// 使用自定义reader而非scanner，增加缓冲区大小提高性能
	reader := bufio.NewReaderSize(resp.Body, 16384) // 增加缓冲区大小到16KB
	
	// 用于存储完整消息的缓冲区
	var messageBuffer bytes.Buffer
	
	// 为每个请求创建独立的状态变量
	accumulatedResponse := ""
	accumulatedReasoning := ""
	
	// 设置监控变量
	totalLines := 0
	validLines := 0
	
	// 创建一个done通道，用于确保goroutine退出
	done := make(chan struct{})
	defer close(done)
	
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
		byteData, isPrefix, err := reader.ReadLine()
		if err != nil {
			if err != io.EOF {
				logError("[reqID:%s] 读取响应出错: %v", reqID, err)
				return err
			} else {
				logDebug("[reqID:%s] 响应流结束", reqID)
			}
			break
		}
		
		// 将字节数据添加到缓冲区
		messageBuffer.Write(byteData)
		
		// 如果数据没有结束，继续读取
		if isPrefix {
			continue
		}
		
		// 获取完整行
		line := messageBuffer.String()
		messageBuffer.Reset()
		
		// 更新行计数
		totalLines++
		
		// 处理数据行
		if strings.HasPrefix(line, "data:") {
			jsonStr := strings.TrimPrefix(line, "data:")
			jsonStr = strings.TrimSpace(jsonStr)
			
			// 特殊处理[DONE]消息
			if jsonStr == "[DONE]" {
				logDebug("[reqID:%s] 收到[DONE]消息", reqID)
				w.Write([]byte("data: [DONE]\n\n"))
				flusher.Flush()
				break
			}
			
			// 解析JSON数据
			var woResp WoCloudResponse
			if err := json.Unmarshal([]byte(jsonStr), &woResp); err != nil {
				logWarn("[reqID:%s] 解析JSON失败: %v, data: %s", reqID, err, jsonStr)
				
				// 尝试修复损坏的JSON
				fixedJson := sanitizeJsonString(jsonStr, reqID)
				
				if err := json.Unmarshal([]byte(fixedJson), &woResp); err != nil {
					logWarn("[reqID:%s] 修复后仍解析失败，跳过此行", reqID)
					continue
				}
				
				logDebug("[reqID:%s] JSON修复成功", reqID)
			}
			
			validLines++
			
			// 检查错误码
			if woResp.Code != 0 {
				logError("[reqID:%s] API返回错误: code=%d, message=%v", reqID, woResp.Code, woResp.Message)
				errorChunk := createErrorChunk(respID, createdTime, fmt.Sprintf("API error: %v", woResp.Message))
				w.Write([]byte("data: " + string(errorChunk) + "\n\n"))
				w.Write([]byte("data: [DONE]\n\n"))
				flusher.Flush()
				return fmt.Errorf("API返回错误码: %d", woResp.Code)
			}
			
			// 处理推理内容
			if woResp.ReasoningContent != "" && woResp.ReasoningContent != accumulatedReasoning {
				// 提取新的推理内容
				newReasoning := woResp.ReasoningContent
				if strings.HasPrefix(woResp.ReasoningContent, accumulatedReasoning) {
					newReasoning = strings.TrimPrefix(woResp.ReasoningContent, accumulatedReasoning)
				}
				
				if newReasoning != "" {
					// 使用sanitizeJsonString清理推理内容
					cleanedReasoning := sanitizeJsonString(newReasoning, reqID)
					
					// 发送推理内容块
					reasoningChunk := createReasoningChunk(respID, createdTime, cleanedReasoning)
					w.Write([]byte("data: " + string(reasoningChunk) + "\n\n"))
					flusher.Flush()
					
					// 更新累积的推理内容
					accumulatedReasoning = woResp.ReasoningContent
					logDebug("[reqID:%s] 发送推理内容块，长度=%d", reqID, len(cleanedReasoning))
				}
			}
			
			// 处理响应内容
			if woResp.Response != "" && woResp.Response != accumulatedResponse {
				// 提取新的响应内容
				newResponse := woResp.Response
				if strings.HasPrefix(woResp.Response, accumulatedResponse) {
					newResponse = strings.TrimPrefix(woResp.Response, accumulatedResponse)
				}
				
				if newResponse != "" {
					// 使用sanitizeJsonString清理响应内容
					cleanedResponse := sanitizeJsonString(newResponse, reqID)
					
					// 发送新的响应内容
					contentChunk := createContentChunk(respID, createdTime, cleanedResponse)
					w.Write([]byte("data: " + string(contentChunk) + "\n\n"))
					flusher.Flush()
					
					// 更新累积的响应内容
					accumulatedResponse = woResp.Response
					logDebug("[reqID:%s] 发送响应内容块，长度=%d", reqID, len(cleanedResponse))
				}
			}
			
			// 检查是否完成
			if woResp.Finish == 1 {
				logDebug("[reqID:%s] 收到完成标志 finish=1", reqID)
				finishReason := "stop"
				doneChunk := createDoneChunk(respID, createdTime, finishReason)
				w.Write([]byte("data: " + string(doneChunk) + "\n\n"))
				w.Write([]byte("data: [DONE]\n\n"))
				flusher.Flush()
				logInfo("[reqID:%s] 流式请求完成, 总行数=%d, 有效行数=%d, 响应长度=%d", 
					reqID, totalLines, validLines, len(accumulatedResponse))
				return nil
			}
		}
	}
	
	// 发送结束信号（如果没有正常结束）
	logInfo("[reqID:%s] 发送结束信号，总行数=%d, 有效行数=%d", reqID, totalLines, validLines)
	finishReason := "stop"
	doneChunk := createDoneChunk(respID, createdTime, finishReason)
	w.Write([]byte("data: " + string(doneChunk) + "\n\n"))
	w.Write([]byte("data: [DONE]\n\n"))
	flusher.Flush()
	
	return nil
}

// 处理非流式请求 - 专注于收集响应内容
func handleNonStreamingRequest(w http.ResponseWriter, r *http.Request, userMessage string, history []WoCloudHistory, token string, reqID string) error {
	logInfo("[reqID:%s] 处理非流式请求", reqID)

	// 构建 WoCloud 请求
	woReq := WoCloudRequest{
		ModelId: 1,
		Input:   userMessage,
		History: history,
	}

	jsonData, err := json.Marshal(woReq)
	if err != nil {
		logError("[reqID:%s] 序列化请求失败: %v", reqID, err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return err
	}

	// 创建请求
	woURL := TargetURL + "/wohome/ai/assistant/query"
	httpReq, err := http.NewRequest("POST", woURL, bytes.NewBuffer(jsonData))
	if err != nil {
		logError("[reqID:%s] 创建请求失败: %v", reqID, err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return err
	}

	// 设置请求头 - 明确要求流式响应
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream") // 明确要求流式响应，以便后续处理
	httpReq.Header.Set("Origin", TargetURL)
	httpReq.Header.Set("Referer", TargetURL+"/h5/wocloud_ai/?modelType=1")
	httpReq.Header.Set("User-Agent", "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1")
	httpReq.Header.Set("X-YP-Access-Token", token)
	httpReq.Header.Set("X-YP-Client-ID", ClientID)

	// 为每个请求创建一个新的HTTP客户端，避免共享连接池导致的干扰
	client := &http.Client{
		Timeout: time.Duration(appConfig.Timeout) * time.Second,
		Transport: &http.Transport{
			MaxIdleConnsPerHost: 100,
			IdleConnTimeout:     90 * time.Second,
		},
	}
	
	// 记录请求细节
	logDebug("[reqID:%s] 非流式请求详情: URL=%s, 用户消息长度=%d, 历史记录数=%d", 
		reqID, woURL, len(userMessage), len(history))
	
	// 使用重试机制
	resp, err := doRequestWithRetry(httpReq, client, appConfig.MaxRetries)
	if err != nil {
		logError("[reqID:%s] 发送请求失败: %v", reqID, err)
		http.Error(w, "Failed to connect to API", http.StatusBadGateway)
		return err
	}
	defer resp.Body.Close()

	// 检查响应状态
	if resp.StatusCode != http.StatusOK {
		logError("[reqID:%s] API返回非200状态码: %d", reqID, resp.StatusCode)
		
		errorInfo, err := handleWoError(resp)
		if err != nil {
			logError("[reqID:%s] 解析错误响应失败: %v", reqID, err)
			http.Error(w, "Failed to parse error response", http.StatusBadGateway)
			return err
		}
		
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(resp.StatusCode)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"error": errorInfo,
		})
		return fmt.Errorf("API返回错误: %s", errorInfo.Message)
	}

	// 读取整个响应体，记录开始时间
	startTime := time.Now()
	logInfo("[reqID:%s] 开始读取响应体...", reqID)
	
	// 增加缓冲区大小，提高大型响应的读取效率
	bodyBytes, err := io.ReadAll(bufio.NewReaderSize(resp.Body, 65536))
	if err != nil {
		logError("[reqID:%s] 读取响应体失败: %v", reqID, err)
		http.Error(w, "Failed to read API response", http.StatusInternalServerError)
		return err
	}
	
	// 记录响应长度和读取时间
	bodyLen := len(bodyBytes)
	readDuration := time.Since(startTime)
	logInfo("[reqID:%s] 收到流式响应，总长度: %d 字节，读取耗时: %v", reqID, bodyLen, readDuration)
	
	// 解析流式响应，提取最后的有效内容
	parsedResponse, err := parseStreamResponseAsNonStream(bodyBytes, reqID)
	if err != nil {
		logError("[reqID:%s] 解析流式响应失败: %v", reqID, err)
		http.Error(w, "Failed to parse streaming response", http.StatusInternalServerError)
		return err
	}
	
// 获取最终响应内容
finalResponse := parsedResponse.Response
	
// 非流请求时，我们不需要思考内容
finalReasoning := ""

logInfo("[reqID:%s] 最终响应内容长度: %d", reqID, len(finalResponse))

// 确保响应不为空
if finalResponse == "" || len(finalResponse) < 10 {
	logWarn("[reqID:%s] 警告: 最终响应为空或过短，尝试使用智能提取", reqID)
	// 尝试智能提取
	extracted := intelligentContentExtraction(bodyBytes, reqID)
	if extracted != "" && len(extracted) > 10 {
		finalResponse = extracted
		logInfo("[reqID:%s] 成功通过智能提取获取响应内容，长度=%d", reqID, len(extracted))
	} else {
		finalResponse = "抱歉，无法获取有效回复。请稍后再试。"
		logWarn("[reqID:%s] 无法获取有效回复，使用默认消息", reqID)
		// 记录解析错误
		atomic.AddInt64(&parseErrorCounter, 1)
	}
}

// 清理响应内容中的问题字符
finalResponse = sanitizeJsonString(finalResponse, reqID)

// 构建 DeepSeek 格式的响应 - 非流请求时不包含reasoning_content
deepSeekResp := CompletionResponse{
	ID:      fmt.Sprintf("chatcmpl-%s", generateRandomID()),
	Object:  "chat.completion",
	Created: time.Now().Unix(),
	Model:   "DeepSeek-R1",
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
				Content:          finalResponse,
				ReasoningContent: finalReasoning, // 非流式请求不使用推理内容
			},
		},
	},
	Usage: struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	}{
		PromptTokens:     len(userMessage),
		CompletionTokens: len(finalResponse),
		TotalTokens:      len(userMessage) + len(finalResponse),
	},
}

// 尝试编码响应前进行验证
testData, err := json.Marshal(deepSeekResp)
if err != nil {
	logError("[reqID:%s] 警告: 响应编码失败: %v", reqID, err)
	// 进一步清理响应或使用默认响应
	deepSeekResp.Choices[0].Message.Content = "抱歉，服务器返回的响应无法正确处理。请稍后再试。"
	deepSeekResp.Choices[0].Message.ReasoningContent = ""
	// 记录解析错误
	atomic.AddInt64(&parseErrorCounter, 1)
} else {
	logDebug("[reqID:%s] 响应验证成功，大小: %d 字节", reqID, len(testData))
}

// 返回响应
w.Header().Set("Content-Type", "application/json")
if err := json.NewEncoder(w).Encode(deepSeekResp); err != nil {
	logError("[reqID:%s] 编码响应失败: %v", reqID, err)
	http.Error(w, "Failed to encode response", http.StatusInternalServerError)
	return err
}

// 记录总处理时间
totalDuration := time.Since(startTime)
logInfo("[reqID:%s] 非流式请求处理完成，总耗时: %v", reqID, totalDuration)

return nil
}

// 生成随机ID
func generateRandomID() string {
// 简化起见，使用时间戳和随机数
return fmt.Sprintf("%d-%d", time.Now().UnixNano(), time.Now().Unix()%1000)
}

// 生成请求ID用于跟踪
func generateRequestID() string {
return fmt.Sprintf("%x", time.Now().UnixNano())
}

// 调试辅助函数 - 保存响应内容到文件
func saveResponseToFile(data []byte, reqID string) {
// 创建logs目录（如果不存在）
logDir := "logs"
if _, err := os.Stat(logDir); os.IsNotExist(err) {
	os.Mkdir(logDir, 0755)
}

// 创建文件名
timestamp := time.Now().Format("20060102_150405")
filename := filepath.Join(logDir, fmt.Sprintf("response_%s_%s.txt", timestamp, reqID[:8]))

// 写入原始内容
if err := os.WriteFile(filename, data, 0644); err != nil {
	logError("[reqID:%s] 保存响应到文件失败: %v", reqID, err)
	return
}

logDebug("[reqID:%s] 已保存响应到文件: %s", reqID, filename)
}

// 保存完整响应内容以便后续分析
func saveFullResponseForAnalysis(bodyBytes []byte, reqID string) {
analyzeDir := "analysis"
os.MkdirAll(analyzeDir, 0755)

// 创建文件名
filename := filepath.Join(analyzeDir, fmt.Sprintf("failed_response_%s.txt", reqID))

// 写入原始内容
if err := os.WriteFile(filename, bodyBytes, 0644); err != nil {
	logError("[reqID:%s] 保存分析文件失败: %v", reqID, err)
	return
}

logInfo("[reqID:%s] 已保存完整响应内容到文件: %s，请检查分析", reqID, filename)
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

// 检查JSON是否有效的辅助函数
func isValidJSON(str string) bool {
var js interface{}
return json.Unmarshal([]byte(str), &js) == nil
}