package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"io"
	"log"
	"math/rand"
	"net/http"
	"net/url"
	"os"
	"path"
	"strings"
	"sync"
	"time"
)

// Config 结构体用于存储命令行参数配置
type Config struct {
	KeyFile   string // API 密钥文件路径
	TargetURL string // 目标 API 基础 URL
	Port      string // 代理服务器监听端口
	Address   string // 代理服务器监听地址
	Password  string // 客户端身份验证密码
}

// parseFlags 解析命令行参数并返回 Config 实例
func parseFlags() *Config {
	cfg := &Config{}
	flag.StringVar(&cfg.KeyFile, "key-file", "", "Path to the API key file")
	flag.StringVar(&cfg.TargetURL, "target-url", "", "Target API base URL")
	flag.StringVar(&cfg.Port, "port", "8080", "Port to listen on")
	flag.StringVar(&cfg.Address, "address", "localhost", "Address to listen on")
	flag.StringVar(&cfg.Password, "password", "", "Password for client authentication")
	flag.Parse()
	return cfg
}

// KeyPool 管理 API 密钥池
type KeyPool struct {
	keys []string   // 密钥列表
	mu   sync.Mutex // 互斥锁，确保线程安全
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
	return &KeyPool{keys: keys}, nil
}

// GetRandomKey 随机返回一个密钥
func (kp *KeyPool) GetRandomKey() string {
	kp.mu.Lock()
	defer kp.mu.Unlock()
	if len(kp.keys) == 0 {
		return ""
	}
	rand.Seed(time.Now().UnixNano())
	return kp.keys[rand.Intn(len(kp.keys))]
}

// ProxyHandler 处理 HTTP 代理请求
type ProxyHandler struct {
	cfg     *Config      // 配置信息
	keyPool *KeyPool     // 密钥池
	client  *http.Client // HTTP 客户端
}

// NewProxyHandler 创建 ProxyHandler 实例
func NewProxyHandler(cfg *Config, keyPool *KeyPool) *ProxyHandler {
	return &ProxyHandler{
		cfg:     cfg,
		keyPool: keyPool,
		client:  &http.Client{},
	}
}

// ServeHTTP 实现 HTTP 处理逻辑
func (ph *ProxyHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// 记录接收到的请求
	log.Printf("[INFO] Received request: %s %s", r.Method, r.URL.String())

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
	ph.keyPool.mu.Lock()
	defer ph.keyPool.mu.Unlock()
	for _, key := range ph.keyPool.keys {
		if !attempted[key] {
			return key
		}
	}
	return ""
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

// maskKey 用于掩码密钥，保护敏感信息
func maskKey(key string) string {
	if len(key) > 4 {
		return key[:4] + "****"
	}
	return "****"
}

// main 函数，启动代理服务器
func main() {
	// 解析配置
	cfg := parseFlags()
	if cfg.KeyFile == "" || cfg.TargetURL == "" || cfg.Password == "" {
		log.Println("[ERROR] Missing required flags: --key-file, --target-url, --password")
		os.Exit(1)
	}
	log.Printf("[INFO] Configuration loaded: KeyFile=%s, TargetURL=%s, Address=%s, Port=%s", 
		cfg.KeyFile, cfg.TargetURL, cfg.Address, cfg.Port)

	// 初始化密钥池
	keyPool, err := NewKeyPool(cfg.KeyFile)
	if err != nil {
		log.Printf("[ERROR] Failed to initialize key pool: %v", err)
		os.Exit(1)
	}

	// 创建代理处理器
	proxyHandler := NewProxyHandler(cfg, keyPool)

	// 启动服务器
	addr := cfg.Address + ":" + cfg.Port
	log.Printf("[INFO] Starting proxy server on %s", addr)
	if err := http.ListenAndServe(addr, proxyHandler); err != nil {
		log.Printf("[ERROR] Failed to start server: %v", err)
		os.Exit(1)
	}
}