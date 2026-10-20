//go:build unit

package handler

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/Wei-Shaw/sub2api/internal/pkg/openai_compat"
	"github.com/Wei-Shaw/sub2api/internal/pkg/tlsfingerprint"
	"github.com/Wei-Shaw/sub2api/internal/server/middleware"
	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

type chatKeepaliveUpstream struct {
	release <-chan struct{}
	mode    string
	calls   atomic.Int32
}

func (u *chatKeepaliveUpstream) Do(req *http.Request, _ string, _ int64, _ int) (*http.Response, error) {
	call := u.calls.Add(1)
	if u.mode == "headers" || u.mode == "retry" && call == 1 {
		select {
		case <-u.release:
		case <-req.Context().Done():
			return nil, req.Context().Err()
		}
	}
	if u.mode == "retry" && call == 1 {
		return &http.Response{StatusCode: http.StatusBadGateway, Header: http.Header{"Content-Type": {"application/json"}}, Body: io.NopCloser(strings.NewReader(`{"error":{"message":"temporary upstream failure"}}`))}, nil
	}
	r, w := io.Pipe()
	go func() {
		defer w.Close()
		select {
		case <-u.release:
		case <-req.Context().Done():
			return
		}
		payload := `{"type":"response.completed","response":{"id":"resp_test","model":"gpt-5.4","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"complete answer"}]}],"usage":{"input_tokens":1,"output_tokens":2}}}`
		if u.mode == "error" {
			payload = `{"type":"response.failed","response":{"id":"resp_test","status":"failed","error":{"code":"invalid_request_error","message":"bad input"}}}`
		}
		_, _ = io.WriteString(w, "data: "+payload+"\n\n")
	}()
	return &http.Response{StatusCode: http.StatusOK, Header: http.Header{"Content-Type": {"text/event-stream"}}, Body: r}, nil
}

func (u *chatKeepaliveUpstream) DoWithTLS(req *http.Request, proxy string, id int64, concurrency int, _ *tlsfingerprint.Profile) (*http.Response, error) {
	return u.Do(req, proxy, id, concurrency)
}

func TestChatCompletionsJSONKeepalive_DeliversBytesBeforeUpstreamCompletes(t *testing.T) {
	gin.SetMode(gin.TestMode)
	for _, mode := range []string{"headers", "body", "error", "retry"} {
		t.Run(mode, func(t *testing.T) {
			release := make(chan struct{})
			var once sync.Once
			unblock := func() { once.Do(func() { close(release) }) }
			defer unblock()
			cfg := &config.Config{RunMode: config.RunModeSimple}
			cfg.Security.URLAllowlist.AllowInsecureHTTP = true
			cfg.Gateway.ChatNonstreamKeepaliveInterval = 1
			cfg.Gateway.MaxAccountSwitches = 1
			repo := &grokCredentialHandlerRepo{accounts: []service.Account{{
				ID: 801, Name: "chat-test", Platform: service.PlatformOpenAI, Type: service.AccountTypeAPIKey,
				Status: service.StatusActive, Schedulable: true, Concurrency: 1, Priority: 1,
				Credentials: map[string]any{"api_key": "test", "base_url": "http://upstream.example"},
				Extra:       map[string]any{openai_compat.ExtraKeyResponsesSupported: true},
			}}}
			if mode == "retry" {
				second := repo.accounts[0]
				second.ID = 802
				second.Priority = 2
				repo.accounts = append(repo.accounts, second)
			}
			billing := service.NewBillingCacheService(nil, nil, nil, nil, nil, nil, cfg, nil)
			defer billing.Stop()
			upstream := &chatKeepaliveUpstream{release: release, mode: mode}
			gateway := service.NewOpenAIGatewayService(repo, nil, nil, nil, nil, nil, nil, cfg, nil, nil,
				service.NewBillingService(cfg, nil), nil, billing, upstream,
				&service.DeferredService{}, nil, nil, nil, nil, nil, nil, nil)
			cache := &concurrencyCacheMock{
				acquireUserSlotFn:    func(context.Context, int64, int, string) (bool, error) { return true, nil },
				acquireAccountSlotFn: func(context.Context, int64, int, string) (bool, error) { return true, nil },
			}
			h := NewOpenAIGatewayHandler(gateway, service.NewConcurrencyService(cache), billing, &service.APIKeyService{}, nil, nil, nil, nil, cfg)
			groupID := int64(901)
			key := &service.APIKey{ID: 902, GroupID: &groupID,
				User:  &service.User{ID: 903, Status: service.StatusActive},
				Group: &service.Group{ID: groupID, Platform: service.PlatformOpenAI, Status: service.StatusActive}}
			router := gin.New()
			router.POST("/v1/chat/completions", func(c *gin.Context) {
				c.Set(string(middleware.ContextKeyAPIKey), key)
				c.Set(string(middleware.ContextKeyUser), middleware.AuthSubject{UserID: key.User.ID, Concurrency: 1})
				h.ChatCompletions(c)
			})
			server := httptest.NewServer(router)
			defer server.Close()
			// Ensure the blocked upstream is released before server shutdown on failure.
			defer unblock()
			client := &http.Client{Timeout: 4 * time.Second}
			resp, err := client.Post(server.URL+"/v1/chat/completions", "application/json",
				strings.NewReader(`{"model":"gpt-5.4","messages":[{"role":"user","content":"hello"}],"stream":false}`))
			require.NoError(t, err, "headers must arrive while upstream is still blocked")
			defer resp.Body.Close()
			first := make([]byte, 2)
			_, err = io.ReadFull(resp.Body, first)
			require.NoError(t, err)
			require.Equal(t, " \n", string(first))
			require.Contains(t, resp.Header.Get("Content-Type"), "application/json")
			unblock()
			rest, err := io.ReadAll(resp.Body)
			require.NoError(t, err)
			body := append(first, rest...)
			require.True(t, json.Valid(body), "heartbeat and final response must form one JSON document")
			if mode == "retry" {
				require.GreaterOrEqual(t, upstream.calls.Load(), int32(2))
			}
			if mode == "error" {
				require.NotEmpty(t, gjson.GetBytes(body, "error.message").String())
			} else {
				require.Equal(t, "complete answer", gjson.GetBytes(body, "choices.0.message.content").String())
			}
		})
	}
}

func TestChatCompletionsJSONKeepalive_LateErrorDoesNotBecomeSSE(t *testing.T) {
	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	stop := service.StartOpenAIJSONKeepalive(c, 5*time.Millisecond)
	defer stop()
	require.Eventually(t, c.Writer.Written, time.Second, time.Millisecond)
	(&OpenAIGatewayHandler{}).handleStreamingAwareErrorWithCode(c, 502, "upstream_error", "read_error", "interrupted", true, true)
	require.True(t, json.Valid(w.Body.Bytes()))
	require.Equal(t, "read_error", gjson.Get(w.Body.String(), "error.code").String())
}
