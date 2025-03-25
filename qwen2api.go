package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"crypto/tls"
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

// 版本和API常量
const (
	Version   = "1.0.1" // 升级版本号
	TargetURL = "https://chat.qwen.ai/api/chat/completions"
	ModelsURL = "https://chat.qwen.ai/api/models"
	FilesURL  = "https://chat.qwen.ai/api/v1/files/"
	TasksURL  = "https://chat.qwen.ai/api/v1/tasks/status/"
)

// 默认模型列表（当获取接口失败时使用）
var DefaultModels = []string{
	"qwen-max-latest",
	"qwen-plus-latest",
	"qwen2.5-vl-72b-instruct",
	"qwen2.5-14b-instruct-1m",
	"qvq-72b-preview",
	"qwq-32b-preview",
	"qwen2.5-coder-32b-instruct",
	"qwen-turbo-latest",
	"qwen2.5-72b-instruct",
}

// 扩展模型变种后缀
var ModelSuffixes = []string{
	"",
	"-thinking",
	"-search",
	"-thinking-search",
	"-draw",
}

// 日志级别常量
const (
	LogLevelDebug = "debug"
	LogLevelInfo  = "info"
	LogLevelWarn  = "warn"
	LogLevelError = "error"
)

// WorkerPool 工作池结构体，用于管理goroutine
type WorkerPool struct {
	taskQueue       chan *Task
	workerCount     int
	shutdownChannel chan struct{}
	wg              sync.WaitGroup
}

// Task 任务结构体，包含请求处理所需数据
type Task struct {
	r        *http.Request
	w        http.ResponseWriter
	done     chan struct{}
	reqID    string
	isStream bool
	apiReq   APIRequest
	path     string
}

// Semaphore 信号量实现，用于限制并发数量
type Semaphore struct {
	sem chan struct{}
}

// 配置结构体
type Config struct {
	Port          string
	Address       string
	LogLevel      string
	DevMode       bool
	DebugEnabled  bool   // 新增调试开关，允许打印敏感信息
	MaxRetries    int
	Timeout       int
	VerifySSL     bool
	WorkerCount   int
	QueueSize     int
	MaxConcurrent int
	APIPrefix     string
	LogDir        string // 日志目录
}

// APIRequest OpenAI兼容的请求结构体
type APIRequest struct {
	Model       string          `json:"model"`
	Messages    []APIMessage    `json:"messages"`
	Stream      bool            `json:"stream"`
	Temperature float64         `json:"temperature,omitempty"`
	MaxTokens   int             `json:"max_tokens,omitempty"`
}

// APIMessage 消息结构体
type APIMessage struct {
	Role          string      `json:"role"`
	Content       interface{} `json:"content"`
	FeatureConfig interface{} `json:"feature_config,omitempty"`
	ChatType      string      `json:"chat_type,omitempty"`
	Extra         interface{} `json:"extra,omitempty"`
}

// 内容项目结构体（处理图像等内容）
type ContentItem struct {
	Type     string    `json:"type,omitempty"`
	Text     string    `json:"text,omitempty"`
	ImageURL *ImageURL `json:"image_url,omitempty"`
	Image    string    `json:"image,omitempty"`
}

// ImageURL 图像URL结构体
type ImageURL struct {
	URL string `json:"url"`
}

// QwenRequest 通义千问API请求结构体
type QwenRequest struct {
	Model             string       `json:"model"`
	Messages          []APIMessage `json:"messages"`
	Stream            bool         `json:"stream"`
	ChatType          string       `json:"chat_type,omitempty"`
	ID                string       `json:"id,omitempty"`
	IncrementalOutput bool         `json:"incremental_output,omitempty"`
	Size              string       `json:"size,omitempty"`
}

// QwenResponse 通义千问API响应结构体 - 更新以匹配实际API响应
type QwenResponse struct {
	Messages []struct {
		Role    string `json:"role"`
		Content string `json:"content"`
		Extra   struct {
			Wanx struct {
				TaskID string `json:"task_id"`
			} `json:"wanx"`
		} `json:"extra"`
	} `json:"messages"`
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
	Error struct {
		Message string `json:"message"`
		Code    string `json:"code"`
	} `json:"error"`
}

// FileUploadResponse 文件上传响应
type FileUploadResponse struct {
	ID string `json:"id"`
}

// TaskStatusResponse 任务状态响应
type TaskStatusResponse struct {
	Content string `json:"content"`
}

// StreamChunk OpenAI兼容的流式响应块
type StreamChunk struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index        int     `json:"index"`
		Delta        struct {
			Role    string `json:"role,omitempty"`
			Content string `json:"content,omitempty"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason,omitempty"`
	} `json:"choices"`
}

// CompletionResponse OpenAI兼容的完成响应
type CompletionResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index        int    `json:"index"`
		Message      struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

// ImagesResponse 图像生成响应
type ImagesResponse struct {
	Created int64      `json:"created"`
	Data    []ImageURL `json:"data"`
}

// ImagesRequest 图像生成请求
type ImagesRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	N      int    `json:"n"`
	Size   string `json:"size"`
}

// ModelData 模型数据
type ModelData struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

// ModelsResponse 模型列表响应
type ModelsResponse struct {
	Object string      `json:"object"`
	Data   []ModelData `json:"data"`
}

// 全局变量
var (
	appConfig    *Config
	logger       *log.Logger
	logLevel     string
	logMutex     sync.Mutex
	workerPool   *WorkerPool
	requestSem   *Semaphore
	requestCount uint64 = 0
	countMutex   sync.Mutex
	
	// 性能指标
	requestCounter   int64
	successCounter   int64
	errorCounter     int64
	avgResponseTime  int64
	queuedRequests   int64
	rejectedRequests int64
)

// NewSemaphore 创建新的信号量
func NewSemaphore(size int) *Semaphore {
	return &Semaphore{
		sem: make(chan struct{}, size),
	}
}

// Acquire 获取信号量（阻塞）
func (s *Semaphore) Acquire() {
	s.sem <- struct{}{}
}

// Release 释放信号量
func (s *Semaphore) Release() {
	<-s.sem
}

// TryAcquire 尝试获取信号量（非阻塞）
func (s *Semaphore) TryAcquire() bool {
	select {
	case s.sem <- struct{}{}:
		return true
	default:
		return false
	}
}

// NewWorkerPool 创建并启动一个新的工作池
func NewWorkerPool(workerCount int, queueSize int) *WorkerPool {
	pool := &WorkerPool{
		taskQueue:       make(chan *Task, queueSize),
		workerCount:     workerCount,
		shutdownChannel: make(chan struct{}),
	}
	
	pool.Start()
	return pool
}

// Start 启动工作池中的worker goroutines
func (pool *WorkerPool) Start() {
	// 启动工作goroutine
	for i := 0; i < pool.workerCount; i++ {
		pool.wg.Add(1)
		go func(workerID int) {
			defer pool.wg.Done()
			
			logInfo("Worker %d 已启动", workerID)
			
			for {
				select {
				case task, ok := <-pool.taskQueue:
					if !ok {
						// 队列已关闭，退出worker
						logInfo("Worker %d 收到队列关闭信号，准备退出", workerID)
						return
					}
					
					logDebug("Worker %d 处理任务 reqID:%s", workerID, task.reqID)
					
					// 处理任务
					switch task.path {
					case "/v1/models":
						handleModels(task.w, task.r)
					case "/v1/chat/completions":
						if task.isStream {
							handleStreamingRequest(task.w, task.r, task.apiReq, task.reqID)
						} else {
							handleNonStreamingRequest(task.w, task.r, task.apiReq, task.reqID)
						}
					case "/v1/images/generations":
						handleImageGenerations(task.w, task.r, task.apiReq, task.reqID)
					}
					
					// 通知任务完成
					close(task.done)
					
				case <-pool.shutdownChannel:
					// 收到关闭信号，退出worker
					logInfo("Worker %d 收到关闭信号，准备退出", workerID)
					return
				}
			}
		}(i)
	}
}

// SubmitTask 提交任务到工作池，非阻塞
func (pool *WorkerPool) SubmitTask(task *Task) (bool, error) {
	select {
	case pool.taskQueue <- task:
		// 任务成功添加到队列
		return true, nil
	default:
		// 队列已满
		return false, fmt.Errorf("任务队列已满")
	}
}

// Shutdown 关闭工作池
func (pool *WorkerPool) Shutdown() {
	logInfo("正在关闭工作池...")
	
	// 发送关闭信号给所有worker
	close(pool.shutdownChannel)
	
	// 等待所有worker退出
	pool.wg.Wait()
	
	// 关闭任务队列
	close(pool.taskQueue)
	
	logInfo("工作池已关闭")
}

// 日志函数
func initLogger(level string) {
	var logOutput io.Writer = os.Stdout
	
	// 如果设置了日志目录，则将日志写入文件
	if appConfig != nil && appConfig.LogDir != "" {
		// 创建日志目录
		if err := os.MkdirAll(appConfig.LogDir, 0755); err != nil {
			fmt.Printf("创建日志目录失败: %v，使用标准输出\n", err)
		} else {
			// 创建日志文件
			logFileName := fmt.Sprintf("%s/qwen-api-%s.log", 
				appConfig.LogDir, time.Now().Format("2006-01-02-15-04-05"))
			logFile, err := os.Create(logFileName)
			if err != nil {
				fmt.Printf("创建日志文件失败: %v，使用标准输出\n", err)
			} else {
				// 使用多输出，同时写入文件和标准输出
				logOutput = io.MultiWriter(os.Stdout, logFile)
				fmt.Printf("日志同时输出到: %s\n", logFileName)
			}
		}
	}
	
	logger = log.New(logOutput, "[QwenAPI] ", log.LstdFlags)
	logLevel = level
}

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

// 解析命令行参数
func parseFlags() *Config {
	cfg := &Config{}
	flag.StringVar(&cfg.Port, "port", "8080", "Port to listen on")
	flag.StringVar(&cfg.Address, "address", "localhost", "Address to listen on")
	flag.StringVar(&cfg.LogLevel, "log-level", LogLevelInfo, "Log level (debug, info, warn, error)")
	flag.BoolVar(&cfg.DevMode, "dev", false, "Enable development mode with enhanced logging")
	flag.BoolVar(&cfg.DebugEnabled, "debug-all", false, "Enable all debug logs including sensitive information")
	flag.IntVar(&cfg.MaxRetries, "max-retries", 3, "Maximum number of retries for failed requests")
	flag.IntVar(&cfg.Timeout, "timeout", 300, "Request timeout in seconds")
	flag.BoolVar(&cfg.VerifySSL, "verify-ssl", true, "Verify SSL certificates")
	flag.IntVar(&cfg.WorkerCount, "workers", 50, "Number of worker goroutines in the pool")
	flag.IntVar(&cfg.QueueSize, "queue-size", 500, "Size of the task queue")
	flag.IntVar(&cfg.MaxConcurrent, "max-concurrent", 100, "Maximum number of concurrent requests")
	flag.StringVar(&cfg.APIPrefix, "api-prefix", "", "API prefix for all endpoints")
	flag.StringVar(&cfg.LogDir, "log-dir", "", "Log directory (if not set, logs to stdout)")
	flag.Parse()
	
	// 如果开发模式开启，自动设置日志级别为debug
	if cfg.DevMode && cfg.LogLevel != LogLevelDebug {
		cfg.LogLevel = LogLevelDebug
		fmt.Println("开发模式已启用，日志级别设置为debug")
	}
	
	// 如果启用了全部调试，将自动启用开发模式
	if cfg.DebugEnabled && !cfg.DevMode {
		cfg.DevMode = true
		cfg.LogLevel = LogLevelDebug
		fmt.Println("全部调试已启用，开发模式和调试日志自动开启")
	}
	
	return cfg
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

// 设置CORS头
func setCORSHeaders(w http.ResponseWriter) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
}

// 生成UUID
func generateUUID() string {
	b := make([]byte, 16)
	_, err := rand.Read(b)
	if err != nil {
		return fmt.Sprintf("%d", time.Now().UnixNano())
	}
	
	return fmt.Sprintf("%x-%x-%x-%x-%x", 
		b[0:4], b[4:6], b[6:8], b[8:10], b[10:])
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

// 请求API的可重试包装函数
func makeAPIRequestWithRetry(ctx context.Context, method, url, authToken string, body []byte, maxRetries int) (*http.Response, error) {
    client := getHTTPClient()
    var resp *http.Response
    var err error
    
    for attempt := 0; attempt <= maxRetries; attempt++ {
        if attempt > 0 {
            logInfo("第%d次重试API请求", attempt)
            time.Sleep(time.Duration(attempt) * 500 * time.Millisecond) // 指数退避
        }
        
        req, err := http.NewRequestWithContext(ctx, method, url, bytes.NewBuffer(body))
        if err != nil {
            continue
        }
        
        // 设置请求头
        req.Header.Set("Content-Type", "application/json")
        req.Header.Set("Authorization", "Bearer "+authToken)
        req.Header.Set("User-Agent", "Mozilla/5.0")
        
        resp, err = client.Do(req)
        if err == nil && resp.StatusCode == http.StatusOK {
            return resp, nil
        }
        
        // 关闭响应体，防止连接泄漏
        if resp != nil && resp.Body != nil {
            resp.Body.Close()
        }
    }
    
    return nil, fmt.Errorf("API请求失败，已重试%d次: %v", maxRetries, err)
}

// 处理API错误的通用函数
func handleAPIError(w http.ResponseWriter, reqID string, statusCode int, body []byte) {
    logError("[reqID:%s] API返回错误，状态码: %d", reqID, statusCode)
    
    // 尝试解析错误响应
    var errorResp struct {
        Error struct {
            Message string `json:"message"`
            Code    string `json:"code"`
        } `json:"error"`
    }
    
    errorMessage := "未知API错误"
    if err := json.Unmarshal(body, &errorResp); err == nil && errorResp.Error.Message != "" {
        errorMessage = errorResp.Error.Message
        logError("[reqID:%s] API错误信息: %s (代码: %s)", reqID, errorResp.Error.Message, errorResp.Error.Code)
    } else {
        // 记录原始响应体以便调试
        logError("[reqID:%s] API错误原始响应: %s", reqID, string(body))
    }
    
    http.Error(w, fmt.Sprintf("API错误: %s", errorMessage), statusCode)
}

// 截断字符串函数，用于日志显示
func truncateString(s string, maxLen int) string {
    if len(s) <= maxLen {
        return s
    }
    return s[:maxLen] + "..."
}

// 主入口函数
func main() {
	// 解析配置
	appConfig = parseFlags()
	
	// 初始化日志
	initLogger(appConfig.LogLevel)

	logInfo("启动服务: 地址=%s, 端口=%s, 版本=%s, 日志级别=%s",
		appConfig.Address, appConfig.Port, Version, appConfig.LogLevel)

	// 创建工作池和信号量
	workerPool = NewWorkerPool(appConfig.WorkerCount, appConfig.QueueSize)
	requestSem = NewSemaphore(appConfig.MaxConcurrent)
	
	logInfo("工作池已创建: %d个worker, 队列大小为%d", appConfig.WorkerCount, appConfig.QueueSize)

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

	// API路径前缀
	apiPrefix := appConfig.APIPrefix

	// 创建处理器
	http.HandleFunc(apiPrefix+"/v1/models", func(w http.ResponseWriter, r *http.Request) {
		setCORSHeaders(w)
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}
		
		// 计数
		countMutex.Lock()
		requestCount++
		currentCount := requestCount
		countMutex.Unlock()
		
		reqID := generateRequestID()
		logInfo("[reqID:%s] 收到模型列表请求 #%d", reqID, currentCount)
		
		// 请求计数
		atomic.AddInt64(&requestCounter, 1)
		
		startTime := time.Now()
		
		// 创建任务
		task := &Task{
			r:        r,
			w:        w,
			done:     make(chan struct{}),
			reqID:    reqID,
			path:     "/v1/models",
		}
		
		// 尝试获取信号量
		if !requestSem.TryAcquire() {
			// 请求数量超过限制
			atomic.AddInt64(&rejectedRequests, 1)
			logWarn("[reqID:%s] 请求被拒绝: 当前并发请求数已达上限", reqID)
			w.Header().Set("Retry-After", "30")
			http.Error(w, "Server is busy, please try again later", http.StatusServiceUnavailable)
			return
		}
		
		// 释放信号量（在函数返回时）
		defer requestSem.Release()
		
		// 添加到任务队列
		atomic.AddInt64(&queuedRequests, 1)
		submitted, err := workerPool.SubmitTask(task)
		if !submitted {
			atomic.AddInt64(&queuedRequests, -1)
			atomic.AddInt64(&rejectedRequests, 1)
			logError("[reqID:%s] 提交任务失败: %v", reqID, err)
			w.Header().Set("Retry-After", "60")
			http.Error(w, "Server queue is full, please try again later", http.StatusServiceUnavailable)
			return
		}
		
		logInfo("[reqID:%s] 任务已提交到队列", reqID)
		
		// 等待任务完成或超时
		select {
		case <-task.done:
			// 任务已完成
			logInfo("[reqID:%s] 任务已完成", reqID)
		case <-r.Context().Done():
			// 请求被取消或超时
			logWarn("[reqID:%s] 请求被取消或超时", reqID)
		}
		
		// 请求处理完成，更新指标
		atomic.AddInt64(&queuedRequests, -1)
		elapsed := time.Since(startTime).Milliseconds()
		
		// 更新平均响应时间
		atomic.AddInt64(&avgResponseTime, elapsed)
		
		if r.Context().Err() == nil {
			// 成功计数增加
			atomic.AddInt64(&successCounter, 1)
			logInfo("[reqID:%s] 请求处理成功，耗时: %dms", reqID, elapsed)
		} else {
			logError("[reqID:%s] 请求处理失败: %v, 耗时: %dms", reqID, r.Context().Err(), elapsed)
		}
	})

	http.HandleFunc(apiPrefix+"/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		setCORSHeaders(w)
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}
		
		// 计数器增加
		countMutex.Lock()
		requestCount++
		currentCount := requestCount
		countMutex.Unlock()
		
		reqID := generateRequestID()
		logInfo("[reqID:%s] 收到新请求 #%d", reqID, currentCount)
		
		// 请求计数
		atomic.AddInt64(&requestCounter, 1)
		
		startTime := time.Now()
		
		// 尝试获取信号量
		if !requestSem.TryAcquire() {
			// 请求数量超过限制
			atomic.AddInt64(&rejectedRequests, 1)
			logWarn("[reqID:%s] 请求 #%d 被拒绝: 当前并发请求数已达上限", reqID, currentCount)
			w.Header().Set("Retry-After", "30")
			http.Error(w, "Server is busy, please try again later", http.StatusServiceUnavailable)
			return
		}
		
		// 释放信号量（在函数返回时）
		defer requestSem.Release()
		
		// 解析请求体
		var apiReq APIRequest
		if err := json.NewDecoder(r.Body).Decode(&apiReq); err != nil {
			logError("[reqID:%s] 解析请求失败: %v", reqID, err)
			http.Error(w, "Invalid request body", http.StatusBadRequest)
			return
		}
		
		// 创建任务
		task := &Task{
			r:        r,
			w:        w,
			done:     make(chan struct{}),
			reqID:    reqID,
			isStream: apiReq.Stream,
			apiReq:   apiReq,
			path:     "/v1/chat/completions",
		}
		
		// 添加到任务队列
		atomic.AddInt64(&queuedRequests, 1)
		submitted, err := workerPool.SubmitTask(task)
		if !submitted {
			atomic.AddInt64(&queuedRequests, -1)
			atomic.AddInt64(&rejectedRequests, 1)
			logError("[reqID:%s] 提交任务失败: %v", reqID, err)
			w.Header().Set("Retry-After", "60")
			http.Error(w, "Server queue is full, please try again later", http.StatusServiceUnavailable)
			return
		}
		
		logInfo("[reqID:%s] 任务已提交到队列", reqID)
		
		// 等待任务完成或超时
		select {
		case <-task.done:
			// 任务已完成
			logInfo("[reqID:%s] 任务已完成", reqID)
		case <-r.Context().Done():
			// 请求被取消或超时
			logWarn("[reqID:%s] 请求被取消或超时", reqID)
		}
		
		// 请求处理完成，更新指标
		atomic.AddInt64(&queuedRequests, -1)
		elapsed := time.Since(startTime).Milliseconds()
		
		// 更新平均响应时间
		atomic.AddInt64(&avgResponseTime, elapsed)
		
		if r.Context().Err() == nil {
			// 成功计数增加
			atomic.AddInt64(&successCounter, 1)
			logInfo("[reqID:%s] 请求处理成功，耗时: %dms", reqID, elapsed)
		} else {
			logError("[reqID:%s] 请求处理失败: %v, 耗时: %dms", reqID, r.Context().Err(), elapsed)
		}
	})
	
	http.HandleFunc(apiPrefix+"/v1/images/generations", func(w http.ResponseWriter, r *http.Request) {
		setCORSHeaders(w)
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}
		
		// 计数器增加
		countMutex.Lock()
		requestCount++
		currentCount := requestCount
		countMutex.Unlock()
		
		reqID := generateRequestID()
		logInfo("[reqID:%s] 收到图像生成请求 #%d", reqID, currentCount)
		
		// 请求计数
		atomic.AddInt64(&requestCounter, 1)
		
		startTime := time.Now()
		
		// 尝试获取信号量
		if !requestSem.TryAcquire() {
			// 请求数量超过限制
			atomic.AddInt64(&rejectedRequests, 1)
			logWarn("[reqID:%s] 请求 #%d 被拒绝: 当前并发请求数已达上限", reqID, currentCount)
			w.Header().Set("Retry-After", "30")
			http.Error(w, "Server is busy, please try again later", http.StatusServiceUnavailable)
			return
		}
		
		// 释放信号量（在函数返回时）
		defer requestSem.Release()
		
		// 解析请求体
		var apiReq APIRequest
		if err := json.NewDecoder(r.Body).Decode(&apiReq); err != nil {
			logError("[reqID:%s] 解析请求失败: %v", reqID, err)
			http.Error(w, "Invalid request body", http.StatusBadRequest)
			return
		}
		
		// 创建任务
		task := &Task{
			r:        r,
			w:        w,
			done:     make(chan struct{}),
			reqID:    reqID,
			apiReq:   apiReq,
			path:     "/v1/images/generations",
		}
		
		// 添加到任务队列
		atomic.AddInt64(&queuedRequests, 1)
		submitted, err := workerPool.SubmitTask(task)
		if !submitted {
			atomic.AddInt64(&queuedRequests, -1)
			atomic.AddInt64(&rejectedRequests, 1)
			logError("[reqID:%s] 提交任务失败: %v", reqID, err)
			w.Header().Set("Retry-After", "60")
			http.Error(w, "Server queue is full, please try again later", http.StatusServiceUnavailable)
			return
		}
		
		logInfo("[reqID:%s] 任务已提交到队列", reqID)
		
		// 等待任务完成或超时
		select {
		case <-task.done:
			// 任务已完成
			logInfo("[reqID:%s] 任务已完成", reqID)
		case <-r.Context().Done():
			// 请求被取消或超时
			logWarn("[reqID:%s] 请求被取消或超时", reqID)
		}
		
		// 请求处理完成，更新指标
		atomic.AddInt64(&queuedRequests, -1)
		elapsed := time.Since(startTime).Milliseconds()
		
		// 更新平均响应时间
		atomic.AddInt64(&avgResponseTime, elapsed)
		
		if r.Context().Err() == nil {
			// 成功计数增加
			atomic.AddInt64(&successCounter, 1)
			logInfo("[reqID:%s] 请求处理成功，耗时: %dms", reqID, elapsed)
		} else {
			logError("[reqID:%s] 请求处理失败: %v, 耗时: %dms", reqID, r.Context().Err(), elapsed)
		}
	})
	
	// 添加健康检查端点
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		setCORSHeaders(w)
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}
		
		// 获取各种计数器的值
		reqCount := atomic.LoadInt64(&requestCounter)
		succCount := atomic.LoadInt64(&successCounter)
		errCount := atomic.LoadInt64(&errorCounter)
		queuedCount := atomic.LoadInt64(&queuedRequests)
		rejectedCount := atomic.LoadInt64(&rejectedRequests)
		
		// 计算平均响应时间
		var avgTime int64 = 0
		if reqCount > 0 {
			avgTime = atomic.LoadInt64(&avgResponseTime) / reqCount
		}
		
		// 构建响应
		stats := map[string]interface{}{
			"status":           "ok",
			"version":          Version,
			"requests":         reqCount,
			"success":          succCount,
			"errors":           errCount,
			"queued":           queuedCount,
			"rejected":         rejectedCount,
			"avg_time_ms":      avgTime,
			"worker_count":     workerPool.workerCount,
			"queue_size":       len(workerPool.taskQueue),
			"queue_capacity":   cap(workerPool.taskQueue),
			"queue_percent":    float64(len(workerPool.taskQueue)) / float64(cap(workerPool.taskQueue)) * 100,
			"concurrent_limit": appConfig.MaxConcurrent,
		}
		
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(stats)
	})
	
	// 添加日志级别动态调整端点
	http.HandleFunc("/debug/log-level", func(w http.ResponseWriter, r *http.Request) {
		setCORSHeaders(w)
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}
		
		// 获取请求的日志级别
		level := r.URL.Query().Get("level")
		if level != "" && (level == LogLevelDebug || level == LogLevelInfo || level == LogLevelWarn || level == LogLevelError) {
			oldLevel := logLevel
			logLevel = level
			w.WriteHeader(http.StatusOK)
			fmt.Fprintf(w, "Log level changed from %s to %s", oldLevel, logLevel)
			logInfo("日志级别已更改为: %s", logLevel)
			return
		}
		
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, "Current log level: %s", logLevel)
	})
	
	// 添加调试开关端点
	http.HandleFunc("/debug/toggle", func(w http.ResponseWriter, r *http.Request) {
		setCORSHeaders(w)
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}
		
		// 切换调试模式
		oldState := appConfig.DebugEnabled
		appConfig.DebugEnabled = !oldState
		
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, "Debug mode changed from %v to %v", oldState, appConfig.DebugEnabled)
		logInfo("调试模式已切换: %v -> %v", oldState, appConfig.DebugEnabled)
	})
	
	// 添加原始响应回显测试端点
	http.HandleFunc("/debug/echo-stream", func(w http.ResponseWriter, r *http.Request) {
		setCORSHeaders(w)
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}
		
		reqID := generateRequestID()
		logInfo("[reqID:%s] 收到调试回显请求", reqID)
		
		// 从请求中提取token
		authToken, err := extractToken(r)
		if err != nil {
			logError("[reqID:%s] 提取token失败: %v", reqID, err)
			http.Error(w, "无效的认证信息", http.StatusUnauthorized)
			return
		}
		
		// 解析请求体
		var apiReq APIRequest
		if err := json.NewDecoder(r.Body).Decode(&apiReq); err != nil {
			logError("[reqID:%s] 解析请求失败: %v", reqID, err)
			http.Error(w, "Invalid request body", http.StatusBadRequest)
			return
		}
		
		// 创建通义千问请求
		qwenReq := QwenRequest{
			Model:    apiReq.Model,
			Messages: apiReq.Messages,
			Stream:   true,
			ChatType: "t2t",
			ID:       generateUUID(),
		}
		
		// 序列化请求
		reqData, err := json.Marshal(qwenReq)
		if err != nil {
			logError("[reqID:%s] 序列化请求失败: %v", reqID, err)
			http.Error(w, "内部服务器错误", http.StatusInternalServerError)
			return
		}
		
		// 创建HTTP请求
		req, err := http.NewRequestWithContext(r.Context(), "POST", TargetURL, bytes.NewBuffer(reqData))
		if err != nil {
			logError("[reqID:%s] 创建请求失败: %v", reqID, err)
			http.Error(w, "内部服务器错误", http.StatusInternalServerError)
			return
		}
		
		// 设置请求头
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+authToken)
		req.Header.Set("User-Agent", "Mozilla/5.0")
		req.Header.Set("Accept", "text/event-stream")
		
		logInfo("[reqID:%s] 调试回显 - API请求数据: %s", reqID, string(reqData))
		
		// 发送请求
		client := getHTTPClient()
		resp, err := client.Do(req)
		if err != nil {
			logError("[reqID:%s] 发送请求失败: %v", reqID, err)
			http.Error(w, "连接到API失败", http.StatusBadGateway)
			return
		}
		defer resp.Body.Close()
		
		// 打印响应头
		logInfo("[reqID:%s] 调试回显 - API响应状态: %d %s", reqID, resp.StatusCode, resp.Status)
		logInfo("[reqID:%s] 调试回显 - API响应头:", reqID)
		for headerName, headerValues := range resp.Header {
			logInfo("[reqID:%s]   %s: %s", reqID, headerName, strings.Join(headerValues, ", "))
		}
		
		// 设置响应头
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		
		// 读取并回显所有响应内容
		responseData, err := io.ReadAll(resp.Body)
		if err != nil {
			logError("[reqID:%s] 读取响应失败: %v", reqID, err)
			http.Error(w, "读取API响应失败", http.StatusInternalServerError)
			return
		}
		
		// 记录并回显完整响应
		logInfo("[reqID:%s] 调试回显 - 完整响应(%d字节):\n%s", reqID, len(responseData), string(responseData))
		w.Write([]byte(fmt.Sprintf("API Response Status: %d %s\n\n", resp.StatusCode, resp.Status)))
		w.Write([]byte("API Response Headers:\n"))
		for headerName, headerValues := range resp.Header {
			w.Write([]byte(fmt.Sprintf("%s: %s\n", headerName, strings.Join(headerValues, ", "))))
		}
		w.Write([]byte("\nAPI Response Body:\n"))
		w.Write(responseData)
		
		// 响应结束
		logInfo("[reqID:%s] 调试回显 - 完成，发送%d字节数据", reqID, len(responseData))
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
	
	// 关闭工作池
	workerPool.Shutdown()
	
	logInfo("Server gracefully stopped")
}

// 生成请求ID
func generateRequestID() string {
	return fmt.Sprintf("%x", time.Now().UnixNano())
}

// 处理模型列表请求
func handleModels(w http.ResponseWriter, r *http.Request) {
	logInfo("处理模型列表请求")

	// 从请求中提取token
	authToken, err := extractToken(r)
	if err != nil {
		logWarn("提取token失败: %v", err)
		// 使用默认模型列表
		returnDefaultModels(w)
		return
	}

	// 请求通义千问API获取模型列表
	client := getHTTPClient()
	req, err := http.NewRequest("GET", ModelsURL, nil)
	if err != nil {
		logError("创建请求失败: %v", err)
		returnDefaultModels(w)
		return
	}

	// 设置请求头
	req.Header.Set("Authorization", "Bearer "+authToken)
	req.Header.Set("User-Agent", "Mozilla/5.0")

	// 发送请求
	resp, err := client.Do(req)
	if err != nil {
		logError("请求模型列表失败: %v", err)
		returnDefaultModels(w)
		return
	}
	defer resp.Body.Close()

	// 检查响应状态
	if resp.StatusCode != http.StatusOK {
		logError("获取模型列表返回非200状态码: %d", resp.StatusCode)
		returnDefaultModels(w)
		return
	}

	// 解析响应
	var qwenResp struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&qwenResp); err != nil {
		logError("解析模型列表响应失败: %v", err)
		returnDefaultModels(w)
		return
	}

	// 提取模型ID
	models := make([]string, 0, len(qwenResp.Data))
	for _, model := range qwenResp.Data {
		models = append(models, model.ID)
	}

	// 如果没有获取到模型，使用默认列表
	if len(models) == 0 {
		logWarn("未获取到模型，使用默认列表")
		returnDefaultModels(w)
		return
	}

	// 扩展模型列表，增加变种后缀
	expandedModels := make([]ModelData, 0, len(models)*len(ModelSuffixes))
	for _, model := range models {
		for _, suffix := range ModelSuffixes {
			expandedModels = append(expandedModels, ModelData{
				ID:      model + suffix,
				Object:  "model",
				Created: time.Now().Unix(),
				OwnedBy: "qwen",
			})
		}
	}

	// 构建响应
	modelsResp := ModelsResponse{
		Object: "list",
		Data:   expandedModels,
	}

	// 返回响应
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(modelsResp)
}

// 返回默认模型列表
func returnDefaultModels(w http.ResponseWriter) {
	// 扩展默认模型列表，增加变种后缀
	expandedModels := make([]ModelData, 0, len(DefaultModels)*len(ModelSuffixes))
	for _, model := range DefaultModels {
		for _, suffix := range ModelSuffixes {
			expandedModels = append(expandedModels, ModelData{
				ID:      model + suffix,
				Object:  "model",
				Created: time.Now().Unix(),
				OwnedBy: "qwen",
			})
		}
	}

	// 构建响应
	modelsResp := ModelsResponse{
		Object: "list",
		Data:   expandedModels,
	}

	// 返回响应
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(modelsResp)
}

// 创建错误响应块
func createErrorChunk(id string, created int64, model string, errorMsg string) []byte {
    errorReason := "error"
    chunk := StreamChunk{
        ID:      id,
        Object:  "chat.completion.chunk",
        Created: created,
        Model:   model,
        Choices: []struct {
            Index        int     `json:"index"`
            Delta        struct {
                Role    string `json:"role,omitempty"`
                Content string `json:"content,omitempty"`
            } `json:"delta"`
            FinishReason *string `json:"finish_reason,omitempty"`
        }{
            {
                Index: 0,
                Delta: struct {
                    Role    string `json:"role,omitempty"`
                    Content string `json:"content,omitempty"`
                }{
                    Content: "【流式处理出错，请重试】",
                },
                FinishReason: &errorReason,
            },
        },
    }

    data, _ := json.Marshal(chunk)
    return data
}

// 处理聊天完成请求（流式）- 重写版本
func handleStreamingRequest(w http.ResponseWriter, r *http.Request, apiReq APIRequest, reqID string) {
    logInfo("[reqID:%s] 处理流式请求", reqID)

    // 从请求中提取token
    authToken, err := extractToken(r)
    if err != nil {
        logError("[reqID:%s] 提取token失败: %v", reqID, err)
        http.Error(w, "无效的认证信息", http.StatusUnauthorized)
        return
    }

    // 检查消息
    if len(apiReq.Messages) == 0 {
        logError("[reqID:%s] 消息为空", reqID)
        http.Error(w, "消息为空", http.StatusBadRequest)
        return
    }

    // 准备模型名和聊天类型
    modelName := "qwen-turbo-latest"
    if apiReq.Model != "" {
        modelName = apiReq.Model
    }
    chatType := "t2t"

    // 处理特殊模型名后缀
    if strings.Contains(modelName, "-draw") {
        handleDrawRequest(w, r, apiReq, reqID, authToken)
        return
    }

    // 处理思考模式
    if strings.Contains(modelName, "-thinking") {
        modelName = strings.Replace(modelName, "-thinking", "", 1)
        lastMsgIdx := len(apiReq.Messages) - 1
        if lastMsgIdx >= 0 {
            apiReq.Messages[lastMsgIdx].FeatureConfig = map[string]interface{}{
                "thinking_enabled": true,
            }
        }
    }

    // 处理搜索模式
    if strings.Contains(modelName, "-search") {
        modelName = strings.Replace(modelName, "-search", "", 1)
        chatType = "search"
        lastMsgIdx := len(apiReq.Messages) - 1
        if lastMsgIdx >= 0 {
            apiReq.Messages[lastMsgIdx].ChatType = "search"
        }
    }

    // 处理图片消息
    lastMsgIdx := len(apiReq.Messages) - 1
    if lastMsgIdx >= 0 {
        lastMsg := apiReq.Messages[lastMsgIdx]
        
        // 检查内容是否为数组
        contentArray, ok := lastMsg.Content.([]interface{})
        if ok {
            // 处理内容数组
            for i, item := range contentArray {
                itemMap, isMap := item.(map[string]interface{})
                if !isMap {
                    continue
                }

                // 检查是否包含图像URL
                if imageURL, hasImageURL := itemMap["image_url"]; hasImageURL {
                    imageURLMap, isMap := imageURL.(map[string]interface{})
                    if !isMap {
                        continue
                    }

                    // 获取URL
                    url, hasURL := imageURLMap["url"].(string)
                    if !hasURL {
                        continue
                    }

                    // 上传图像
                    imageID, uploadErr := uploadImage(url, authToken)
                    if uploadErr != nil {
                        logError("[reqID:%s] 上传图像失败: %v", reqID, uploadErr)
                        continue
                    }

                    // 替换内容
                    contentArrayCopy := make([]interface{}, len(contentArray))
                    copy(contentArrayCopy, contentArray)
                    contentArrayCopy[i] = map[string]interface{}{
                        "type":  "image",
                        "image": imageID,
                    }
                    apiReq.Messages[lastMsgIdx].Content = contentArrayCopy
                    break
                }
            }
        }
    }

    // 创建通义千问请求
    qwenReq := QwenRequest{
        Model:    modelName,
        Messages: apiReq.Messages,
        Stream:   true,
        ChatType: chatType,
        ID:       generateUUID(),
    }

    // 序列化请求
    reqData, err := json.Marshal(qwenReq)
    if err != nil {
        logError("[reqID:%s] 序列化请求失败: %v", reqID, err)
        http.Error(w, "内部服务器错误", http.StatusInternalServerError)
        return
    }

    // 打印请求数据
    logInfo("[reqID:%s] API请求数据: %s", reqID, string(reqData))

    // 创建HTTP请求
    req, err := http.NewRequestWithContext(r.Context(), "POST", TargetURL, bytes.NewBuffer(reqData))
    if err != nil {
        logError("[reqID:%s] 创建请求失败: %v", reqID, err)
        http.Error(w, "内部服务器错误", http.StatusInternalServerError)
        return
    }

    // 设置请求头
    req.Header.Set("Content-Type", "application/json")
    req.Header.Set("Authorization", "Bearer "+authToken)
    req.Header.Set("User-Agent", "Mozilla/5.0")
    req.Header.Set("Accept", "text/event-stream")
    
    // 打印所有请求头
    logInfo("[reqID:%s] API请求头:", reqID)
    for headerName, headerValues := range req.Header {
        logInfo("[reqID:%s]   %s: %s", reqID, headerName, strings.Join(headerValues, ", "))
    }

    // 发送请求 - 直接使用HTTP客户端，不使用重试函数以便更好调试
    client := getHTTPClient()
    startTime := time.Now()
    resp, err := client.Do(req)
    requestDuration := time.Since(startTime)
    
    if err != nil {
        logError("[reqID:%s] 发送请求失败: %v", reqID, err)
        http.Error(w, "连接到API失败", http.StatusBadGateway)
        return
    }
    defer resp.Body.Close()

    // 打印响应状态和头
    logInfo("[reqID:%s] API响应状态: %d %s (耗时: %v)", reqID, resp.StatusCode, resp.Status, requestDuration)
    logInfo("[reqID:%s] API响应头:", reqID)
    for headerName, headerValues := range resp.Header {
        logInfo("[reqID:%s]   %s: %s", reqID, headerName, strings.Join(headerValues, ", "))
    }

    // 检查响应状态
    if resp.StatusCode != http.StatusOK {
        bodyBytes, _ := io.ReadAll(resp.Body)
        logError("[reqID:%s] API返回非200状态码: %d, 响应: %s", reqID, resp.StatusCode, string(bodyBytes))
        http.Error(w, fmt.Sprintf("API错误，状态码: %d", resp.StatusCode), resp.StatusCode)
        return
    }

    // 设置响应头
    w.Header().Set("Content-Type", "text/event-stream")
    w.Header().Set("Cache-Control", "no-cache")
    w.Header().Set("Connection", "keep-alive")
    w.Header().Set("X-Accel-Buffering", "no") // 禁用Nginx缓冲

    // 创建响应ID和时间戳
    respID := fmt.Sprintf("chatcmpl-%s", generateUUID())
    createdTime := time.Now().Unix()

    // 创建读取器和Flusher
    reader := bufio.NewReaderSize(resp.Body, 16384)
    flusher, ok := w.(http.Flusher)
    if !ok {
        logError("[reqID:%s] 流式传输不支持", reqID)
        http.Error(w, "流式传输不支持", http.StatusInternalServerError)
        return
    }

    // 发送角色块
    roleChunk := createRoleChunk(respID, createdTime, modelName)
    if _, err := w.Write([]byte("data: " + string(roleChunk) + "\n\n")); err != nil {
        logError("[reqID:%s] 写入角色块失败: %v", reqID, err)
        return
    }
    flusher.Flush()
    logInfo("[reqID:%s] 已发送角色块", reqID)

    // 用于去重和累积的变量
    previousContent := ""
    accumulatedContent := ""
    buffer := ""
    totalBytesReceived := 0
    chunkCount := 0

    // 持续读取响应 - 逐行读取而不是使用块读取
    logInfo("[reqID:%s] 开始读取流式响应", reqID)
    
    // 直接读取并记录完整响应
    respBody, err := io.ReadAll(resp.Body)
    if err != nil {
        logError("[reqID:%s] 读取完整响应失败: %v", reqID, err)
        http.Error(w, "读取API响应失败", http.StatusInternalServerError)
        return
    }
    
    // 记录完整响应内容
    logInfo("[reqID:%s] 完整响应内容(%d字节):\n%s", reqID, len(respBody), string(respBody))
    
    // 重新构建响应供后续处理
    resp.Body = io.NopCloser(bytes.NewReader(respBody))
    reader = bufio.NewReaderSize(resp.Body, 16384)()

    // 逐行读取响应
    for {
        // 添加超时检测
        select {
        case <-r.Context().Done():
            logWarn("[reqID:%s] 请求超时或被客户端取消", reqID)
            return
        default:
            // 继续处理
        }

        // 读取一行数据 - 使用ReadLine而不是Read
        line, isPrefix, err := reader.ReadLine()
        if err != nil {
            if err != io.EOF {
                logError("[reqID:%s] 读取响应出错: %v", reqID, err)
                
                // 发送错误通知给客户端
                errorChunk := createErrorChunk(respID, createdTime, modelName, "读取响应出错")
                w.Write([]byte("data: " + string(errorChunk) + "\n\n"))
                w.Write([]byte("data: [DONE]\n\n"))
                flusher.Flush()
                return
            }
            break
        }

        // 处理长行
        lineData := string(line)
        for isPrefix {
            var lineContinuation []byte
            lineContinuation, isPrefix, err = reader.ReadLine()
            if err != nil {
                if err != io.EOF {
                    logError("[reqID:%s] 读取长行剩余部分失败: %v", reqID, err)
                }
                break
            }
            lineData += string(lineContinuation)
        }

        // 记录每一行
        chunkCount++
        totalBytesReceived += len(lineData)
        logInfo("[reqID:%s] 收到第%d块数据(%d字节): %s", reqID, chunkCount, len(lineData), lineData)

        // 如果是空行，表示一个SSE消息的边界
        if len(lineData) == 0 {
            if len(buffer) > 0 {
                // 处理缓冲的SSE消息
                logInfo("[reqID:%s] 处理SSE消息: %s", reqID, buffer)
                
                // 判断是否是data:前缀
                if strings.HasPrefix(buffer, "data:") {
                    dataStr := strings.TrimPrefix(buffer, "data:")
                    dataStr = strings.TrimSpace(dataStr)
                    
                    // 处理[DONE]消息
                    if dataStr == "[DONE]" {
                        logInfo("[reqID:%s] 收到[DONE]消息", reqID)
                        w.Write([]byte("data: [DONE]\n\n"))
                        flusher.Flush()
                        buffer = ""
                        continue
                    }
                    
                    // 尝试解析JSON
                    var qwenResp QwenResponse
                    if err := json.Unmarshal([]byte(dataStr), &qwenResp); err != nil {
                        logWarn("[reqID:%s] 解析JSON失败: %v, data: %s", reqID, err, dataStr)
                        buffer = ""
                        continue
                    }
                    
                    // 提取内容
                    hasContent := false
                    for _, choice := range qwenResp.Choices {
                        content := choice.Delta.Content
                        if content == "" {
                            continue
                        }
                        
                        hasContent = true
                        logInfo("[reqID:%s] 成功解析内容: %s", reqID, content)
                        
                        // 改进去重逻辑
                        var newContent string
                        if previousContent != "" && strings.HasPrefix(content, previousContent) {
                            // 提取新增部分
                            newContent = content[len(previousContent):]
                            if newContent == "" {
                                logInfo("[reqID:%s] 内容重复，跳过", reqID)
                                continue
                            }
                            logInfo("[reqID:%s] 去重后新增内容: %s", reqID, newContent)
                        } else {
                            // 使用完整内容
                            newContent = content
                        }
                        
                        // 创建内容块并发送
                        contentChunk := createContentChunk(respID, createdTime, modelName, newContent)
                        if _, err := w.Write([]byte("data: " + string(contentChunk) + "\n\n")); err != nil {
                            logError("[reqID:%s] 写入内容块失败: %v", reqID, err)
                            return
                        }
                        flusher.Flush()
                        
                        // 更新前一个内容和累积内容
                        previousContent = content
                        accumulatedContent += newContent
                    }
                    
                    // 处理完成标志
                    for _, choice := range qwenResp.Choices {
                        if choice.FinishReason != "" {
                            finishReason := choice.FinishReason
                            logInfo("[reqID:%s] 收到完成标志: %s", reqID, finishReason)
                            doneChunk := createDoneChunk(respID, createdTime, modelName, finishReason)
                            if _, err := w.Write([]byte("data: " + string(doneChunk) + "\n\n")); err != nil {
                                logError("[reqID:%s] 写入完成块失败: %v", reqID, err)
                                return
                            }
                            flusher.Flush()
                        }
                    }
                    
                    // 调试日志：记录是否有内容
                    if !hasContent {
                        logInfo("[reqID:%s] 该消息没有内容", reqID)
                    }
                }
                
                // 清空缓冲区，准备下一个SSE消息
                buffer = ""
            }
        } else {
            // 累积非空行到缓冲区
            if len(buffer) > 0 {
                buffer += "\n"
            }
            buffer += lineData
        }
    }

    // 记录累积的内容
    logInfo("[reqID:%s] 流处理完成，收到%d块数据，总计%d字节，累积内容长度: %d", 
        reqID, chunkCount, totalBytesReceived, len(accumulatedContent))
    
    // 确保发送结束信号
    if _, err := w.Write([]byte("data: [DONE]\n\n")); err != nil {
        logError("[reqID:%s] 写入结束信号失败: %v", reqID, err)
    }
    flusher.Flush()
}

// 处理聊天完成请求（非流式）
func handleNonStreamingRequest(w http.ResponseWriter, r *http.Request, apiReq APIRequest, reqID string) {
	logInfo("[reqID:%s] 处理非流式请求", reqID)

	// 从请求中提取token
	authToken, err := extractToken(r)
	if err != nil {
		logError("[reqID:%s] 提取token失败: %v", reqID, err)
		http.Error(w, "无效的认证信息", http.StatusUnauthorized)
		return
	}

	// 检查消息
	if len(apiReq.Messages) == 0 {
		logError("[reqID:%s] 消息为空", reqID)
		http.Error(w, "消息为空", http.StatusBadRequest)
		return
	}

	// 准备模型名和聊天类型
	modelName := "qwen-turbo-latest"
	if apiReq.Model != "" {
		modelName = apiReq.Model
	}
	chatType := "t2t"

	// 处理特殊模型名后缀
	if strings.Contains(modelName, "-draw") {
		handleDrawRequest(w, r, apiReq, reqID, authToken)
		return
	}

	// 处理思考模式
	if strings.Contains(modelName, "-thinking") {
		modelName = strings.Replace(modelName, "-thinking", "", 1)
		lastMsgIdx := len(apiReq.Messages) - 1
		if lastMsgIdx >= 0 {
			apiReq.Messages[lastMsgIdx].FeatureConfig = map[string]interface{}{
				"thinking_enabled": true,
			}
		}
	}

	// 处理搜索模式
	if strings.Contains(modelName, "-search") {
		modelName = strings.Replace(modelName, "-search", "", 1)
		chatType = "search"
		lastMsgIdx := len(apiReq.Messages) - 1
		if lastMsgIdx >= 0 {
			apiReq.Messages[lastMsgIdx].ChatType = "search"
		}
	}

	// 处理图片消息
	lastMsgIdx := len(apiReq.Messages) - 1
	if lastMsgIdx >= 0 {
		lastMsg := apiReq.Messages[lastMsgIdx]
		
		// 检查内容是否为数组
		contentArray, ok := lastMsg.Content.([]interface{})
		if ok {
			// 处理内容数组
			for i, item := range contentArray {
				itemMap, isMap := item.(map[string]interface{})
				if !isMap {
					continue
				}

				// 检查是否包含图像URL
				if imageURL, hasImageURL := itemMap["image_url"]; hasImageURL {
					imageURLMap, isMap := imageURL.(map[string]interface{})
					if !isMap {
						continue
					}

					// 获取URL
					url, hasURL := imageURLMap["url"].(string)
					if !hasURL {
						continue
					}

					// 上传图像
					imageID, uploadErr := uploadImage(url, authToken)
					if uploadErr != nil {
						logError("[reqID:%s] 上传图像失败: %v", reqID, uploadErr)
						continue
					}

					// 替换内容
					contentArrayCopy := make([]interface{}, len(contentArray))
					copy(contentArrayCopy, contentArray)
					contentArrayCopy[i] = map[string]interface{}{
						"type":  "image",
						"image": imageID,
					}
					apiReq.Messages[lastMsgIdx].Content = contentArrayCopy
					break
				}
			}
		}
	}

	// 创建通义千问请求 - 通过流式请求来获取非流式响应
	qwenReq := QwenRequest{
		Model:    modelName,
		Messages: apiReq.Messages,
		Stream:   true, // 使用流式API
		ChatType: chatType,
		ID:       generateUUID(),
	}

	// 序列化请求
	reqData, err := json.Marshal(qwenReq)
	if err != nil {
		logError("[reqID:%s] 序列化请求失败: %v", reqID, err)
		http.Error(w, "内部服务器错误", http.StatusInternalServerError)
		return
	}

	// 创建HTTP请求
	resp, err := makeAPIRequestWithRetry(r.Context(), "POST", TargetURL, authToken, reqData, appConfig.MaxRetries)
	if err != nil {
		logError("[reqID:%s] 发送请求失败: %v", reqID, err)
		http.Error(w, "连接到API失败", http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	// 检查响应状态
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		handleAPIError(w, reqID, resp.StatusCode, bodyBytes)
		return
	}

	// 从流式响应中提取完整内容
	fullContent, err := extractFullContentFromStream(resp.Body, reqID)
	if err != nil {
		logError("[reqID:%s] 提取内容失败: %v", reqID, err)
		http.Error(w, "解析响应失败", http.StatusInternalServerError)
		return
	}

	// 创建非流式响应
	completionResponse := CompletionResponse{
		ID:      fmt.Sprintf("chatcmpl-%s", generateUUID()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   modelName,
		Choices: []struct {
			Index        int    `json:"index"`
			Message      struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		}{
			{
				Index: 0,
				Message: struct {
					Role    string `json:"role"`
					Content string `json:"content"`
				}{
					Role:    "assistant",
					Content: fullContent,
				},
				FinishReason: "stop",
			},
		},
		Usage: struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		}{
			PromptTokens:     estimateTokens(apiReq.Messages),
			CompletionTokens: len(fullContent) / 4,
			TotalTokens:      estimateTokens(apiReq.Messages) + len(fullContent)/4,
		},
	}

	// 返回响应
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(completionResponse)
}

// 处理图像生成请求
func handleImageGenerations(w http.ResponseWriter, r *http.Request, apiReq APIRequest, reqID string) {
	logInfo("[reqID:%s] 处理图像生成请求", reqID)

	// 从请求中提取token
	authToken, err := extractToken(r)
	if err != nil {
		logError("[reqID:%s] 提取token失败: %v", reqID, err)
		http.Error(w, "无效的认证信息", http.StatusUnauthorized)
		return
	}

	// 解析图像生成请求
	var imgReq ImagesRequest
	if err := json.NewDecoder(r.Body).Decode(&imgReq); err != nil {
		logError("[reqID:%s] 解析图像请求失败: %v", reqID, err)
		http.Error(w, "无效的请求体", http.StatusBadRequest)
		return
	}

	// 默认值设置
	if imgReq.Model == "" {
		imgReq.Model = "qwen-max-latest-draw"
	}
	if imgReq.Size == "" {
		imgReq.Size = "1024*1024"
	}
	if imgReq.N <= 0 {
		imgReq.N = 1
	}

	// 获取纯模型名（去除-draw后缀）
	modelName := strings.Replace(imgReq.Model, "-draw", "", 1)
	modelName = strings.Replace(modelName, "-thinking", "", 1)
	modelName = strings.Replace(modelName, "-search", "", 1)

	// 创建图像生成任务
	qwenReq := QwenRequest{
		Stream:            false,
		IncrementalOutput: true,
		ChatType:          "t2i",
		Model:             modelName,
		Messages: []APIMessage{
			{
				Role:     "user",
				Content:  imgReq.Prompt,
				ChatType: "t2i",
				Extra:    map[string]interface{}{},
				FeatureConfig: map[string]interface{}{
					"thinking_enabled": false,
				},
			},
		},
		ID:   generateUUID(),
		Size: imgReq.Size,
	}

	// 序列化请求
	reqData, err := json.Marshal(qwenReq)
	if err != nil {
		logError("[reqID:%s] 序列化请求失败: %v", reqID, err)
		http.Error(w, "内部服务器错误", http.StatusInternalServerError)
		return
	}

	// 创建HTTP请求
	resp, err := makeAPIRequestWithRetry(r.Context(), "POST", TargetURL, authToken, reqData, appConfig.MaxRetries)
	if err != nil {
		logError("[reqID:%s] 发送请求失败: %v", reqID, err)
		http.Error(w, "连接到API失败", http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	// 检查响应状态
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		handleAPIError(w, reqID, resp.StatusCode, bodyBytes)
		return
	}

	// 解析响应获取任务ID
	var qwenResp QwenResponse
	if err := json.NewDecoder(resp.Body).Decode(&qwenResp); err != nil {
		logError("[reqID:%s] 解析响应失败: %v", reqID, err)
		http.Error(w, "解析响应失败", http.StatusInternalServerError)
		return
	}

	// 提取任务ID
	taskID := ""
	for _, msg := range qwenResp.Messages {
		if msg.Role == "assistant" && msg.Extra.Wanx.TaskID != "" {
			taskID = msg.Extra.Wanx.TaskID
			break
		}
	}

	if taskID == "" {
		logError("[reqID:%s] 无法获取图像生成任务ID", reqID)
		http.Error(w, "无法获取图像生成任务ID", http.StatusInternalServerError)
		return
	}

	// 轮询等待图像生成完成
	var imageURL string
	for i := 0; i < 30; i++ {
		select {
		case <-r.Context().Done():
			logWarn("[reqID:%s] 请求超时或被客户端取消", reqID)
			http.Error(w, "请求超时", http.StatusGatewayTimeout)
			return
		default:
			// 继续处理
		}

		// 检查任务状态
		statusURL := TasksURL + taskID
		statusReq, err := http.NewRequestWithContext(r.Context(), "GET", statusURL, nil)
		if err != nil {
			logError("[reqID:%s] 创建状态请求失败: %v", reqID, err)
			time.Sleep(6 * time.Second)
			continue
		}

		// 设置请求头
		statusReq.Header.Set("Authorization", "Bearer "+authToken)
		statusReq.Header.Set("User-Agent", "Mozilla/5.0")

		// 发送请求
		statusResp, err := getHTTPClient().Do(statusReq)
		if err != nil {
			logError("[reqID:%s] 发送状态请求失败: %v", reqID, err)
			time.Sleep(6 * time.Second)
			continue
		}

		// 解析响应
		var statusData TaskStatusResponse
		if err := json.NewDecoder(statusResp.Body).Decode(&statusData); err != nil {
			logError("[reqID:%s] 解析状态响应失败: %v", reqID, err)
			statusResp.Body.Close()
			time.Sleep(6 * time.Second)
			continue
		}
		statusResp.Body.Close()

		// 检查是否有内容
		if statusData.Content != "" {
			imageURL = statusData.Content
			break
		}

		time.Sleep(6 * time.Second)
	}

	if imageURL == "" {
		logError("[reqID:%s] 图像生成超时", reqID)
		http.Error(w, "图像生成超时", http.StatusGatewayTimeout)
		return
	}

	// 构造图像列表
	images := make([]ImageURL, imgReq.N)
	for i := 0; i < imgReq.N; i++ {
		images[i] = ImageURL{URL: imageURL}
	}

	// 返回响应
	imgResp := ImagesResponse{
		Created: time.Now().Unix(),
		Data:    images,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(imgResp)
}

// 处理特殊的绘图请求
func handleDrawRequest(w http.ResponseWriter, r *http.Request, apiReq APIRequest, reqID string, authToken string) {
	logInfo("[reqID:%s] 处理绘图请求", reqID)

	// 获取绘图提示
	var prompt string
	if len(apiReq.Messages) > 0 {
		lastMsg := apiReq.Messages[len(apiReq.Messages)-1]
		prompt, _ = lastMsg.Content.(string)
	}
	
	if prompt == "" {
		logError("[reqID:%s] 绘图提示为空", reqID)
		http.Error(w, "绘图提示为空", http.StatusBadRequest)
		return
	}

	// 准备绘图请求参数
	size := "1024*1024"
	modelName := strings.Replace(apiReq.Model, "-draw", "", 1)
	modelName = strings.Replace(modelName, "-thinking", "", 1)
	modelName = strings.Replace(modelName, "-search", "", 1)

	// 创建绘图请求
	qwenReq := QwenRequest{
		Stream:            false,
		IncrementalOutput: true,
		ChatType:          "t2i",
		Model:             modelName,
		Messages: []APIMessage{
			{
				Role:     "user",
				Content:  prompt,
				ChatType: "t2i",
				Extra:    map[string]interface{}{},
				FeatureConfig: map[string]interface{}{
					"thinking_enabled": false,
				},
			},
		},
		ID:   generateUUID(),
		Size: size,
	}

	// 序列化请求
	reqData, err := json.Marshal(qwenReq)
	if err != nil {
		logError("[reqID:%s] 序列化请求失败: %v", reqID, err)
		http.Error(w, "内部服务器错误", http.StatusInternalServerError)
		return
	}

	// 创建HTTP请求
	resp, err := makeAPIRequestWithRetry(r.Context(), "POST", TargetURL, authToken, reqData, appConfig.MaxRetries)
	if err != nil {
		logError("[reqID:%s] 发送请求失败: %v", reqID, err)
		http.Error(w, "连接到API失败", http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	// 检查响应状态
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		handleAPIError(w, reqID, resp.StatusCode, bodyBytes)
		return
	}

	// 解析响应获取任务ID
	var qwenResp QwenResponse
	if err := json.NewDecoder(resp.Body).Decode(&qwenResp); err != nil {
		logError("[reqID:%s] 解析响应失败: %v", reqID, err)
		http.Error(w, "解析响应失败", http.StatusInternalServerError)
		return
	}

	// 提取任务ID
	taskID := ""
	for _, msg := range qwenResp.Messages {
		if msg.Role == "assistant" && msg.Extra.Wanx.TaskID != "" {
			taskID = msg.Extra.Wanx.TaskID
			break
		}
	}

	if taskID == "" {
		logError("[reqID:%s] 无法获取图像生成任务ID", reqID)
		http.Error(w, "无法获取图像生成任务ID", http.StatusInternalServerError)
		return
	}

	// 轮询等待图像生成完成
	var imageURL string
	for i := 0; i < 30; i++ {
		select {
		case <-r.Context().Done():
			logWarn("[reqID:%s] 请求超时或被客户端取消", reqID)
			http.Error(w, "请求超时", http.StatusGatewayTimeout)
			return
		default:
			// 继续处理
		}

		// 检查任务状态
		statusURL := TasksURL + taskID
		statusReq, err := http.NewRequestWithContext(r.Context(), "GET", statusURL, nil)
		if err != nil {
			logError("[reqID:%s] 创建状态请求失败: %v", reqID, err)
			time.Sleep(6 * time.Second)
			continue
		}

		// 设置请求头
		statusReq.Header.Set("Authorization", "Bearer "+authToken)
		statusReq.Header.Set("User-Agent", "Mozilla/5.0")

		// 发送请求
		statusResp, err := getHTTPClient().Do(statusReq)
		if err != nil {
			logError("[reqID:%s] 发送状态请求失败: %v", reqID, err)
			time.Sleep(6 * time.Second)
			continue
		}

		// 解析响应
		var statusData TaskStatusResponse
		if err := json.NewDecoder(statusResp.Body).Decode(&statusData); err != nil {
			logError("[reqID:%s] 解析状态响应失败: %v", reqID, err)
			statusResp.Body.Close()
			time.Sleep(6 * time.Second)
			continue
		}
		statusResp.Body.Close()

		// 检查是否有内容
		if statusData.Content != "" {
			imageURL = statusData.Content
			break
		}

		time.Sleep(6 * time.Second)
	}

	if imageURL == "" {
		logError("[reqID:%s] 图像生成超时", reqID)
		http.Error(w, "图像生成超时", http.StatusGatewayTimeout)
		return
	}

	// 返回OpenAI标准格式响应（使用Markdown嵌入图片）
	completionResponse := CompletionResponse{
		ID:      fmt.Sprintf("chatcmpl-%s", generateUUID()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   apiReq.Model,
		Choices: []struct {
			Index        int    `json:"index"`
			Message      struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		}{
			{
				Index: 0,
				Message: struct {
					Role    string `json:"role"`
					Content string `json:"content"`
				}{
					Role:    "assistant",
					Content: fmt.Sprintf("![%s](%s)", imageURL, imageURL),
				},
				FinishReason: "stop",
			},
		},
		Usage: struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		}{
			PromptTokens:     1024,
			CompletionTokens: 1024,
			TotalTokens:      2048,
		},
	}

	// 返回响应
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(completionResponse)
}

// 从流式响应中提取完整内容 - 重写版
func extractFullContentFromStream(body io.ReadCloser, reqID string) (string, error) {
    var contentBuilder strings.Builder
    reader := bufio.NewReaderSize(body, 16384)
    buffer := ""

    logInfo("[reqID:%s] 开始从流中提取完整内容", reqID)

    for {
        // 读取一块数据
        chunk := make([]byte, 4096)
        n, err := reader.Read(chunk)
        if err != nil {
            if err != io.EOF {
                return contentBuilder.String(), fmt.Errorf("读取流数据出错: %v", err)
            }
            break
        }

        // 添加到缓冲区并记录调试信息
        newData := string(chunk[:n])
        buffer += newData
        logDebug("[reqID:%s] 提取内容 - 收到原始数据(%d字节): %s", reqID, n, truncateString(newData, 100))

        // 使用双换行分割SSE消息
        messages := strings.Split(buffer, "\n\n")
        
        // 保留最后一个可能不完整的消息
        if len(messages) > 0 {
            buffer = messages[len(messages)-1]
            
            // 处理所有完整的消息
            for i := 0; i < len(messages)-1; i++ {
                message := messages[i]
                if !strings.HasPrefix(message, "data: ") {
                    continue
                }

                // 提取数据部分
                dataStr := strings.TrimPrefix(message, "data: ")
                dataStr = strings.TrimSpace(dataStr)

                // 处理[DONE]消息
                if dataStr == "[DONE]" {
                    logDebug("[reqID:%s] 提取内容 - 收到[DONE]消息", reqID)
                    continue
                }

                // 解析JSON
                var qwenResp QwenResponse
                if err := json.Unmarshal([]byte(dataStr), &qwenResp); err != nil {
                    logWarn("[reqID:%s] 提取内容 - 解析JSON失败: %v, data: %s", reqID, err, truncateString(dataStr, 100))
                    continue
                }

                // 提取内容增量
                for _, choice := range qwenResp.Choices {
                    if choice.Delta.Content != "" {
                        contentBuilder.WriteString(choice.Delta.Content)
                        logDebug("[reqID:%s] 提取内容 - 累积增量: %s", reqID, truncateString(choice.Delta.Content, 50))
                    }
                }
            }
        }
    }

    // 记录提取的内容长度
    contentStr := contentBuilder.String()
    logInfo("[reqID:%s] 非流式模式：成功提取完整内容，长度: %d", reqID, len(contentStr))
    return contentStr, nil
}

// 上传图像到千问API
func uploadImage(base64Data string, authToken string) (string, error) {
	// 从base64数据中提取图片数据
	if !strings.HasPrefix(base64Data, "data:") {
		return "", fmt.Errorf("invalid base64 data format")
	}
	
	parts := strings.SplitN(base64Data, ",", 2)
	if len(parts) != 2 {
		return "", fmt.Errorf("invalid base64 data format")
	}
	
	imageData, err := base64.StdEncoding.DecodeString(parts[1])
	if err != nil {
		return "", fmt.Errorf("failed to decode base64 data: %v", err)
	}
	
	// 创建multipart表单
	body := bytes.Buffer{}
	writer := multipart.NewWriter(&body)
	
	// 添加文件
	part, err := writer.CreateFormFile("file", fmt.Sprintf("image-%d.jpg", time.Now().UnixNano()))
	if err != nil {
		return "", fmt.Errorf("failed to create form file: %v", err)
	}
	
	if _, err := part.Write(imageData); err != nil {
		return "", fmt.Errorf("failed to write image data: %v", err)
	}
	
	// 关闭writer
	if err := writer.Close(); err != nil {
		return "", fmt.Errorf("failed to close writer: %v", err)
	}
	
	// 创建HTTP请求
	req, err := http.NewRequest("POST", FilesURL, &body)
	if err != nil {
		return "", fmt.Errorf("failed to create request: %v", err)
	}
	
	// 设置请求头
	req.Header.Set("Content-Type", writer.FormDataContentType())
	req.Header.Set("Authorization", "Bearer "+authToken)
	req.Header.Set("User-Agent", "Mozilla/5.0")
	
	// 发送请求
	client := getHTTPClient()
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to send request: %v", err)
	}
	defer resp.Body.Close()
	
	// 检查响应状态
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("API returned non-200 status code: %d, response: %s", resp.StatusCode, string(bodyBytes))
	}
	
	// 解析响应
	var uploadResp FileUploadResponse
	if err := json.NewDecoder(resp.Body).Decode(&uploadResp); err != nil {
		return "", fmt.Errorf("failed to parse response: %v", err)
	}
	
	return uploadResp.ID, nil
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
			Delta        struct {
				Role    string `json:"role,omitempty"`
				Content string `json:"content,omitempty"`
			} `json:"delta"`
			FinishReason *string `json:"finish_reason,omitempty"`
		}{
			{
				Index: 0,
				Delta: struct {
					Role    string `json:"role,omitempty"`
					Content string `json:"content,omitempty"`
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
			Delta        struct {
				Role    string `json:"role,omitempty"`
				Content string `json:"content,omitempty"`
			} `json:"delta"`
			FinishReason *string `json:"finish_reason,omitempty"`
		}{
			{
				Index: 0,
				Delta: struct {
					Role    string `json:"role,omitempty"`
					Content string `json:"content,omitempty"`
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
func createDoneChunk(id string, created int64, model string, reason string) []byte {
	finishReason := reason
	chunk := StreamChunk{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   model,
		Choices: []struct {
			Index        int     `json:"index"`
			Delta        struct {
				Role    string `json:"role,omitempty"`
				Content string `json:"content,omitempty"`
			} `json:"delta"`
			FinishReason *string `json:"finish_reason,omitempty"`
		}{
			{
				Index:        0,
				Delta:        struct {
					Role    string `json:"role,omitempty"`
					Content string `json:"content,omitempty"`
				}{},
				FinishReason: &finishReason,
			},
		},
	}

	data, _ := json.Marshal(chunk)
	return data
}

// 估算tokens（简单实现）
func estimateTokens(messages []APIMessage) int {
	var total int
	for _, msg := range messages {
		switch content := msg.Content.(type) {
		case string:
			total += len(content) / 4
		case []interface{}:
			for _, item := range content {
				if itemMap, ok := item.(map[string]interface{}); ok {
					if text, ok := itemMap["text"].(string); ok {
						total += len(text) / 4
					}
				}
			}
		}
	}
	return total
}