package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"path"
	"strings"
	"sync"
)

// Config 结构体用于存储命令行参数配置
type Config struct {
	KeyFile    string // API 密钥文件路径
	TargetURL  string // 目标 API 基础 URL
	Port       string // 代理服务器监听端口
	Address    string // 代理服务器监听地址
	Password   string // 客户端身份验证密码
	MaxWorkers int    // 最大工作协程数
	MaxQueue   int    // 最大请求队列长度
}

// parseFlags 解析命令行参数并返回 Config 实例
func parseFlags() *Config {
	cfg := &Config{}
	
	// 基本配置
	flag.StringVar(&cfg.KeyFile, "key-file", "", "Path to the API key file")
	flag.StringVar(&cfg.TargetURL, "target-url", "", "Target API base URL")
	flag.StringVar(&cfg.Port, "port", "8080", "Port to listen on")
	flag.StringVar(&cfg.Address, "address", "localhost", "Address to listen on")
	flag.StringVar(&cfg.Password, "password", "", "Password for client authentication")
	
	// WorkerPool相关配置，直接存储到Config结构体字段中
	flag.IntVar(&cfg.MaxWorkers, "max-workers", 50, "Maximum number of worker goroutines")
	flag.IntVar(&cfg.MaxQueue, "max-queue", 500, "Maximum size of request queue")
	
	// 添加帮助信息
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage of %s:\n", os.Args[0])
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExample:\n  %s --key-file=./keys.txt --target-url=https://api.example.com --password=mysecret --max-workers=100 --max-queue=1000\n", os.Args[0])
	}
	
	flag.Parse()
	
	// 验证参数值范围
	if cfg.MaxWorkers <= 0 {
		log.Printf("[WARN] Invalid max-workers value %d, using default 50", cfg.MaxWorkers)
		cfg.MaxWorkers = 50
	}
	
	if cfg.MaxQueue <= 0 {
		log.Printf("[WARN] Invalid max-queue value %d, using default 500", cfg.MaxQueue)
		cfg.MaxQueue = 500
	}
	
	return cfg
}

// KeyPool 管理 API 密钥池
type KeyPool struct {
	keys         []string   // 密钥列表
	mu           sync.Mutex // 互斥锁，确保线程安全
	currentIndex int        // 当前密钥索引，用于循环抽取
}

// NewKeyPool 从文件中加载密钥并创建 KeyPool 实例
func NewKeyPool(filePath string) (*KeyPool, error) {
	file, err := os.Open(filePath)
	if err != nil {
		log.Printf("[ERROR] Failed to open key file %s: %v", filePath, err)
		return nil, err
	}
	defer file.Close()

	var keys []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		key := strings.TrimSpace(scanner.Text())
		if key != "" {
			keys = append(keys, key)
		}
	}
	if err := scanner.Err(); err != nil {
		log.Printf("[ERROR] Failed to read key file %s: %v", filePath, err)
		return nil, err
	}
	log.Printf("[INFO] Loaded %d keys from file %s", len(keys), filePath)
	return &KeyPool{keys: keys, currentIndex: 0}, nil
}

// GetRandomKey 按顺序循环返回一个密钥
func (kp *KeyPool) GetRandomKey() string {
	kp.mu.Lock()
	defer kp.mu.Unlock()
	if len(kp.keys) == 0 {
		return ""
	}
	key := kp.keys[kp.currentIndex]
	kp.currentIndex = (kp.currentIndex + 1) % len(kp.keys) // 循环到下一个索引
	return key
}

// 定义请求结构体
type ProxyRequest struct {
	Request  *http.Request
	Response http.ResponseWriter
	Done     chan bool // 用于通知请求处理完成
}

// Worker结构体，表示一个工作协程
type Worker struct {
	ID         int
	TaskQueue  chan *ProxyRequest // 任务队列
	Quit       chan bool          // 退出信号
	WorkerPool *WorkerPool        // 所属工作池
}

// 创建新的Worker
func NewWorker(id int, workerPool *WorkerPool) *Worker {
	return &Worker{
		ID:         id,
		TaskQueue:  make(chan *ProxyRequest),
		Quit:       make(chan bool),
		WorkerPool: workerPool,
	}
}

// Worker开始工作
func (w *Worker) Start() {
	go func() {
		for {
			// 将worker注册到工作池的空闲队列
			w.WorkerPool.WorkerQueue <- w.TaskQueue

			select {
			case task := <-w.TaskQueue:
				// 处理请求
				w.WorkerPool.HandleFunc(task.Response, task.Request)
				task.Done <- true
			case <-w.Quit:
				// 收到退出信号
				return
			}
		}
	}()
}

// Worker停止工作
func (w *Worker) Stop() {
	go func() {
		w.Quit <- true
	}()
}

// WorkerPool结构体，管理工作协程池
type WorkerPool struct {
	WorkerQueue chan chan *ProxyRequest // 空闲Worker队列
	TaskQueue   chan *ProxyRequest      // 任务队列
	MaxWorkers  int                     // 最大Worker数量
	MaxQueue    int                     // 最大队列长度
	HandleFunc  func(http.ResponseWriter, *http.Request) // 请求处理函数
}

// 创建新的WorkerPool
func NewWorkerPool(maxWorkers int, maxQueue int, handleFunc func(http.ResponseWriter, *http.Request)) *WorkerPool {
	pool := &WorkerPool{
		WorkerQueue: make(chan chan *ProxyRequest, maxWorkers),
		TaskQueue:   make(chan *ProxyRequest, maxQueue),
		MaxWorkers:  maxWorkers,
		MaxQueue:    maxQueue,
		HandleFunc:  handleFunc,
	}
	return pool
}

// 启动WorkerPool
func (wp *WorkerPool) Start() {
	// 创建并启动workers
	for i := 0; i < wp.MaxWorkers; i++ {
		worker := NewWorker(i, wp)
		worker.Start()
		log.Printf("[INFO] Started worker %d", i)
	}

	// 启动任务分发协程
	go wp.dispatch()
}

// 停止WorkerPool
func (wp *WorkerPool) Stop() {
	// TODO: 实现停止逻辑
}

// 将任务分发给空闲worker
func (wp *WorkerPool) dispatch() {
	for {
		select {
		case task := <-wp.TaskQueue:
			// 等待空闲worker
			workerTaskQueue := <-wp.WorkerQueue
			// 将任务发送给worker
			workerTaskQueue <- task
		}
	}
}

// 将请求提交到WorkerPool
func (wp *WorkerPool) Submit(response http.ResponseWriter, request *http.Request) bool {
	task := &ProxyRequest{
		Request:  request,
		Response: response,
		Done:     make(chan bool, 1),
	}

	select {
	case wp.TaskQueue <- task:
		// 请求成功加入队列
		<-task.Done // 等待任务完成
		return true
	default:
		// 队列已满，实现背压
		log.Println("[WARN] Task queue is full, rejecting request")
		http.Error(response, "Server is busy, please try again later", http.StatusServiceUnavailable)
		return false
	}
}

// ProxyHandler 处理 HTTP 代理请求
type ProxyHandler struct {
	cfg        *Config      // 配置信息
	keyPool    *KeyPool     // 密钥池
	client     *http.Client // HTTP 客户端
	workerPool *WorkerPool  // 工作协程池
}

// NewProxyHandler 创建 ProxyHandler 实例
func NewProxyHandler(cfg *Config, keyPool *KeyPool) *ProxyHandler {
	handler := &ProxyHandler{
		cfg:     cfg,
		keyPool: keyPool,
		client:  &http.Client{},
	}
	return handler
}

// InitWorkerPool 初始化工作协程池
func (ph *ProxyHandler) InitWorkerPool(maxWorkers int, maxQueue int) {
	ph.workerPool = NewWorkerPool(maxWorkers, maxQueue, ph.HandleRequest)
	ph.workerPool.Start()
	log.Printf("[INFO] Started worker pool with %d workers and queue size %d", maxWorkers, maxQueue)
}

// ServeHTTP 实现 HTTP 处理逻辑
func (ph *ProxyHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// 记录接收到的请求
	log.Printf("[INFO] Received request: %s %s", r.Method, r.URL.String())

	// 将请求提交到工作池处理
	ph.workerPool.Submit(w, r)
}

// HandleRequest 处理请求的方法，由Worker调用
func (ph *ProxyHandler) HandleRequest(w http.ResponseWriter, r *http.Request) {
	// 验证客户端身份
	if !ph.authenticate(r) {
		log.Println("[WARN] Unauthorized access attempt")
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}
	log.Println("[INFO] Authentication successful")

	// 尝试解析请求体中的模型信息
	model, err := ph.extractModelFromRequest(r)
	if err != nil {
		log.Printf("[WARN] Failed to extract model from request: %v", err)
	} else if model != "" {
		log.Printf("[INFO] Model specified in request: %s", model)
	}

	// 构建目标 URL
	targetURL, err := ph.buildTargetURL(r)
	if err != nil {
		log.Printf("[ERROR] Failed to build target URL: %v", err)
		http.Error(w, "Bad Request", http.StatusBadRequest)
		return
	}
	log.Printf("[INFO] Target URL: %s", targetURL)

	// 重试逻辑
	maxRetries := len(ph.keyPool.keys)
	attemptedKeys := make(map[string]bool)
	log.Printf("[INFO] Starting key selection process, total keys available: %d", maxRetries)

	for i := 0; i < maxRetries; i++ {
		key := ph.getUnusedKey(attemptedKeys)
		if key == "" {
			log.Printf("[ERROR] No unused keys remaining after %d attempts", i)
			break
		}
		attemptedKeys[key] = true
		maskedKey := maskKey(key)
		log.Printf("[INFO] Attempt %d/%d: Selecting key %s", i+1, maxRetries, maskedKey)

		// 创建请求
		req, err := ph.createRequest(r, targetURL, key)
		if err != nil {
			log.Printf("[ERROR] Failed to create request with key %s: %v", maskedKey, err)
			log.Printf("[INFO] Switching to another key due to request creation failure")
			continue
		}

		// 发送请求
		log.Printf("[INFO] Sending request to target API with key %s", maskedKey)
		resp, err := ph.client.Do(req)
		if err != nil {
			log.Printf("[ERROR] Failed to send request with key %s: %v", maskedKey, err)
			log.Printf("[INFO] Switching to another key due to network error")
			continue
		}
		defer resp.Body.Close()

		// 处理响应
		log.Printf("[INFO] Received response with status code %d", resp.StatusCode)
		if resp.StatusCode >= 200 && resp.StatusCode < 300 {
			log.Println("[INFO] Request successful, forwarding response")
			ph.forwardResponse(w, resp)
			return
		} else if resp.StatusCode == 403 || resp.StatusCode == 429 {
			log.Printf("[WARN] Received %d status code with key %s", resp.StatusCode, maskedKey)
			log.Printf("[INFO] Switching to another key due to status code %d", resp.StatusCode)
			continue
		} else {
			log.Printf("[INFO] Forwarding response with status code %d", resp.StatusCode)
			ph.forwardResponse(w, resp)
			return
		}
	}

	// 所有密钥尝试后仍失败
	log.Printf("[ERROR] All %d keys failed after retries", maxRetries)
	http.Error(w, "Failed to get response from API after all retries", http.StatusBadGateway)
}

// getUnusedKey 获取一个未使用过的密钥
func (ph *ProxyHandler) getUnusedKey(attempted map[string]bool) string {
	key := ph.keyPool.GetRandomKey()
	// 如果获取到的密钥已使用过，则尝试其他密钥
	for attempted[key] && len(attempted) < len(ph.keyPool.keys) {
		key = ph.keyPool.GetRandomKey()
	}
	// 如果所有密钥都已尝试过，返回空字符串
	if attempted[key] {
		return ""
	}
	return key
}

// authenticate 验证客户端身份
func (ph *ProxyHandler) authenticate(r *http.Request) bool {
	authHeader := r.Header.Get("Authorization")
	if authHeader == "" {
		return false
	}
	parts := strings.Split(authHeader, " ")
	if len(parts) != 2 || parts[0] != "Bearer" {
		return false
	}
	return parts[1] == ph.cfg.Password
}

// buildTargetURL 构建目标 API 的完整 URL
func (ph *ProxyHandler) buildTargetURL(r *http.Request) (string, error) {
	u, err := url.Parse(ph.cfg.TargetURL)
	if err != nil {
		return "", err
	}
	u.Path = path.Join(u.Path, r.URL.Path)
	u.RawQuery = r.URL.RawQuery
	return u.String(), nil
}

// createRequest 创建转发请求
func (ph *ProxyHandler) createRequest(r *http.Request, targetURL, key string) (*http.Request, error) {
	req, err := http.NewRequest(r.Method, targetURL, r.Body)
	if err != nil {
		return nil, err
	}

	// 复制并修改请求头
	for k, v := range r.Header {
		if k != "Host" && k != "Connection" && k != "Proxy-Connection" && k != "Authorization" {
			req.Header[k] = v
		}
	}
	req.Header.Set("Authorization", "Bearer "+key)
	return req, nil
}

// forwardResponse 将响应转发给客户端，支持流式和非流式
func (ph *ProxyHandler) forwardResponse(w http.ResponseWriter, resp *http.Response) {
	// 设置响应头
	for k, v := range resp.Header {
		w.Header()[k] = v
	}
	w.WriteHeader(resp.StatusCode)

	// 处理流式响应
	if strings.Contains(resp.Header.Get("Content-Type"), "text/event-stream") || resp.Header.Get("Transfer-Encoding") == "chunked" {
		log.Println("[INFO] Handling streaming response")
		flusher, ok := w.(http.Flusher)
		if !ok {
			log.Println("[ERROR] Streaming unsupported by server")
			http.Error(w, "Streaming unsupported", http.StatusInternalServerError)
			return
		}
		reader := bufio.NewReader(resp.Body)
		for {
			line, err := reader.ReadBytes('\n')
			if err != nil {
				if err == io.EOF {
					log.Println("[INFO] Stream ended")
					break
				}
				log.Printf("[ERROR] Error reading stream: %v", err)
				http.Error(w, "Error reading stream", http.StatusInternalServerError)
				return
			}
			w.Write(line)
			flusher.Flush()
		}
	} else {
		// 非流式响应，直接复制
		_, err := io.Copy(w, resp.Body)
		if err != nil {
			log.Printf("[ERROR] Failed to forward response: %v", err)
		}
	}
}

// extractModelFromRequest 尝试从请求体中提取模型名称
func (ph *ProxyHandler) extractModelFromRequest(r *http.Request) (string, error) {
	if r.Body == nil {
		return "", nil
	}
	body, err := io.ReadAll(r.Body)
	if err != nil {
		return "", err
	}
	r.Body = io.NopCloser(strings.NewReader(string(body)))

	var data map[string]interface{}
	if err := json.Unmarshal(body, &data); err != nil {
		return "", err
	}
	if model, ok := data["model"].(string); ok {
		return model, nil
	}
	return "", nil
}

// maskKey 直接返回原始密钥，不再进行掩码处理
func maskKey(key string) string {
	return key
}

// main 函数，启动代理服务器
func main() {
	// 解析配置
	cfg := parseFlags()
	if cfg.KeyFile == "" || cfg.TargetURL == "" || cfg.Password == "" {
		log.Println("[ERROR] Missing required flags: --key-file, --target-url, --password")
		flag.Usage()
		os.Exit(1)
	}
	
	// 输出实际使用的配置参数
	log.Printf("[INFO] Starting with configuration:")
	log.Printf("[INFO] - KeyFile: %s", cfg.KeyFile)
	log.Printf("[INFO] - TargetURL: %s", cfg.TargetURL)
	log.Printf("[INFO] - Address: %s", cfg.Address)
	log.Printf("[INFO] - Port: %s", cfg.Port)
	log.Printf("[INFO] - MaxWorkers: %d", cfg.MaxWorkers)
	log.Printf("[INFO] - MaxQueue: %d", cfg.MaxQueue)

	// 初始化密钥池
	keyPool, err := NewKeyPool(cfg.KeyFile)
	if err != nil {
		log.Printf("[ERROR] Failed to initialize key pool: %v", err)
		os.Exit(1)
	}

	// 创建代理处理器
	proxyHandler := NewProxyHandler(cfg, keyPool)
	
	// 初始化并启动工作池
	proxyHandler.InitWorkerPool(cfg.MaxWorkers, cfg.MaxQueue)

	// 启动服务器
	addr := cfg.Address + ":" + cfg.Port
	log.Printf("[INFO] Starting proxy server on %s", addr)
	if err := http.ListenAndServe(addr, proxyHandler); err != nil {
		log.Printf("[ERROR] Failed to start server: %v", err)
		os.Exit(1)
	}
}