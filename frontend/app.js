// AI Chat 应用
class ChatApp {
    constructor() {
        this.apiUrl = 'http://localhost:5000/api';
        this.selectedModel = '';
        this.chatHistory = [];
        
        this.initElements();
        this.initEventListeners();
        this.loadModels();
        this.loadSystemInfo();
        
        console.log('✅ ChatApp 初始化完成');
    }

    initElements() {
        // 侧边栏
        this.sidebar = document.getElementById('sidebar');
        this.menuToggle = document.getElementById('menuToggle');
        this.newChatBtn = document.getElementById('newChatBtn');
        this.modelSelect = document.getElementById('modelSelect');
        this.deviceStatus = document.getElementById('deviceStatus');
        this.modelStatus = document.getElementById('modelStatus');
        
        // 主界面
        this.messagesContainer = document.getElementById('messagesContainer');
        this.welcomeScreen = document.getElementById('welcomeScreen');
        this.messages = document.getElementById('messages');
        this.messageInput = document.getElementById('messageInput');
        this.sendBtn = document.getElementById('sendBtn');
        
        // 聊天历史
        this.todayChats = document.getElementById('todayChats');
    }

    initEventListeners() {
        // 侧边栏
        this.menuToggle.addEventListener('click', () => this.toggleSidebar());
        this.newChatBtn.addEventListener('click', () => this.createNewChat());
        this.modelSelect.addEventListener('change', (e) => this.selectModel(e.target.value));
        
        // 发送消息
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // 自动调整输入框高度
        this.messageInput.addEventListener('input', () => {
            this.messageInput.style.height = 'auto';
            this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 200) + 'px';
            this.updateSendButton();
        });
        
        // 建议卡片
        document.querySelectorAll('.suggestion-card').forEach(card => {
            card.addEventListener('click', () => {
                const prompt = card.dataset.prompt;
                this.messageInput.value = prompt;
                this.messageInput.focus();
                this.updateSendButton();
            });
        });
    }

    updateSendButton() {
        const hasText = this.messageInput.value.trim().length > 0;
        const hasModel = this.selectedModel !== '';
        this.sendBtn.disabled = !(hasText && hasModel);
    }

    async loadModels() {
        console.log('🔄 开始加载模型列表...');
        console.log('📡 API地址:', this.apiUrl + '/models');
        
        try {
            const response = await fetch(this.apiUrl + '/models');
            console.log('📥 响应状态:', response.status);
            
            if (!response.ok) {
                throw new Error('HTTP ' + response.status);
            }
            
            const data = await response.json();
            console.log('📦 收到数据:', data);
            
            this.modelSelect.innerHTML = '<option value="">选择模型...</option>';
            
            if (!data.models || data.models.length === 0) {
                throw new Error('没有可用的模型');
            }
            
            data.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.id;
                option.textContent = model.icon + ' ' + model.name;
                this.modelSelect.appendChild(option);
            });
            
            console.log('✅ 模型列表加载成功，共', data.models.length, '个模型');
            this.showNotification('加载了 ' + data.models.length + ' 个模型', 'success');
            
        } catch (error) {
            console.error('❌ 加载模型失败:', error);
            
            this.modelSelect.innerHTML = '<option value="">连接失败，请检查后端</option>';
            
            let errorMsg = '无法连接到后端服务';
            if (error.message.includes('Failed to fetch')) {
                errorMsg = '后端服务未启动或端口错误';
            } else if (error.message.includes('CORS')) {
                errorMsg = 'CORS跨域问题';
            } else {
                errorMsg = error.message;
            }
            
            this.showNotification(errorMsg, 'error');
            
            // 5秒后重试
            setTimeout(() => {
                console.log('🔄 5秒后重试...');
                this.loadModels();
            }, 5000);
        }
    }

    async loadSystemInfo() {
        try {
            const response = await fetch(this.apiUrl + '/health');
            const data = await response.json();
            
            this.deviceStatus.textContent = data.cuda_available ? 'GPU' : 'CPU';
            this.modelStatus.textContent = data.loaded_models.length > 0 ?
                data.loaded_models.length + ' 个已加载' : '未加载';
        } catch (error) {
            console.error('加载系统信息失败:', error);
        }
    }

    selectModel(modelId) {
        this.selectedModel = modelId;
        this.updateSendButton();
        
        if (modelId) {
            console.log('✅ 已选择模型:', modelId);
            this.showNotification('模型已选择', 'success');
        }
    }

    async sendMessage() {
        const message = this.messageInput.value.trim();
        
        if (!message || !this.selectedModel) return;

        // 隐藏欢迎界面
        this.welcomeScreen.style.display = 'none';
        
        // 添加用户消息
        this.addMessage('user', message);
        this.messageInput.value = '';
        this.messageInput.style.height = 'auto';
        this.updateSendButton();
        
        // 禁用输入
        this.sendBtn.disabled = true;
        this.messageInput.disabled = true;
        
        // 显示typing指示器
        const typingId = this.showTyping();

        try {
            const response = await fetch(this.apiUrl + '/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    model: this.selectedModel,
                    message: message,
                    history: this.chatHistory
                })
            });

            const data = await response.json();
            
            // 移除typing指示器
            this.removeTyping(typingId);
            
            if (response.ok) {
                if (data.jailbreak_detected) {
                    // 被拦截的消息
                    this.addMessage('assistant', data.response, {
                        jailbreak_detected: true,
                        detection_result: data.detection_result
                    });
                } else {
                    // 正常回复
                    this.addMessage('assistant', data.response, {
                        time: data.generation_time,
                        tokens: data.tokens_generated
                    });
                    
                    this.chatHistory.push(
                        { role: 'user', content: message },
                        { role: 'assistant', content: data.response }
                    );
                }
            } else {
                this.showNotification(data.error || '请求失败', 'error');
            }
        } catch (error) {
            this.removeTyping(typingId);
            console.error('❌ 发送消息失败:', error);
            this.showNotification('网络错误，请检查后端服务', 'error');
        } finally {
            this.sendBtn.disabled = false;
            this.messageInput.disabled = false;
            this.messageInput.focus();
            this.updateSendButton();
        }
    }

    addMessage(role, content, meta) {
        meta = meta || {};
        
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message ' + role;
        
        const isBlocked = meta.jailbreak_detected;
        
        if (isBlocked) {
            messageDiv.classList.add('blocked');
        }
        
        const avatar = role === 'user' ? '👤' : '🤖';
        
        let contentHtml = '<div class="message-text">' + this.escapeHtml(content) + '</div>';
        
        // 如果是被拦截的消息，添加检测详情
        if (isBlocked && meta.detection_result) {
            const result = meta.detection_result;
            
            contentHtml += '<div class="detection-details">';
            contentHtml += '<h4>🔍 安全检测报告</h4>';
            
            // 风险评分 - 添加安全检查
            let riskLevel = 'medium';  // 默认值
            if (result.risk_level) {
                riskLevel = result.risk_level.toLowerCase();
            }
            
            // 风险等级显示名称
            const riskLevelNames = {
                'safe': '安全',
                'low': '低风险',
                'medium': '中等风险',
                'high': '高风险',
                'critical': '严重风险'
            };
            
            const riskScore = result.risk_score || 0;
            const riskName = riskLevelNames[riskLevel] || '未知';
            
            contentHtml += `<div class="risk-score ${riskLevel}">`;
            contentHtml += `风险评分: ${riskScore}/100 `;
            contentHtml += `<span class="risk-badge risk-${riskLevel}">${riskName}</span>`;
            contentHtml += '</div>';
            
            // 检测到的特征
            if (result.matched_features && result.matched_features.length > 0) {
                contentHtml += '<h5>⚠️ 检测到的问题：</h5>';
                contentHtml += '<ul class="feature-list">';
                result.matched_features.slice(0, 5).forEach(feature => {
                    contentHtml += '<li>';
                    contentHtml += `<div class="feature-name">${this.escapeHtml(feature.name || '未知特征')}</div>`;
                    contentHtml += `<div class="feature-desc">${this.escapeHtml(feature.description || '无描述')}</div>`;
                    contentHtml += '</li>';
                });
                contentHtml += '</ul>';
            }
            
            // 可疑关键词
            if (result.suspicious_keywords && result.suspicious_keywords.length > 0) {
                contentHtml += '<h5>🔍 可疑关键词：</h5>';
                contentHtml += '<ul>';
                result.suspicious_keywords.forEach(kw => {
                    const keyword = kw.keyword || '';
                    const name = kw.name || '未知';
                    contentHtml += `<li><code>${this.escapeHtml(keyword)}</code> - ${this.escapeHtml(name)}</li>`;
                });
                contentHtml += '</ul>';
            }
            
            // 建议
            if (result.recommendations && result.recommendations.length > 0) {
                contentHtml += '<div class="recommendations">';
                contentHtml += '<h5>💡 建议：</h5>';
                contentHtml += '<ul>';
                result.recommendations.forEach(rec => {
                    contentHtml += `<li>${this.escapeHtml(rec)}</li>`;
                });
                contentHtml += '</ul>';
                contentHtml += '</div>';
            }
            
            contentHtml += '</div>';
        }
        
        // 普通元信息
        let metaHtml = '';
        if (meta.time && !isBlocked) {
            metaHtml = '<div class="message-meta">';
            metaHtml += '<span>⏱️ ' + meta.time + '秒</span>';
            if (meta.tokens) {
                metaHtml += '<span>⚡ ' + meta.tokens + ' tokens</span>';
            }
            metaHtml += '</div>';
        }
        
        messageDiv.innerHTML = 
            '<div class="message-avatar">' + avatar + '</div>' +
            '<div class="message-content">' +
                contentHtml +
                metaHtml +
            '</div>';
        
        this.messages.appendChild(messageDiv);
        this.scrollToBottom();
    }
        


    showTyping() {
        const typingDiv = document.createElement('div');
        const typingId = 'typing-' + Date.now();
        typingDiv.id = typingId;
        typingDiv.className = 'message assistant';
        typingDiv.innerHTML = 
            '<div class="message-avatar">' +
                '<i class="fas fa-robot"></i>' +
            '</div>' +
            '<div class="message-content">' +
                '<div class="typing-indicator">' +
                    '<div class="typing-dot"></div>' +
                    '<div class="typing-dot"></div>' +
                    '<div class="typing-dot"></div>' +
                '</div>' +
            '</div>';
        
        this.messages.appendChild(typingDiv);
        this.scrollToBottom();
        return typingId;
    }

    removeTyping(typingId) {
        const typingDiv = document.getElementById(typingId);
        if (typingDiv) {
            typingDiv.remove();
        }
    }

    createNewChat() {
        this.chatHistory = [];
        this.messages.innerHTML = '';
        this.welcomeScreen.style.display = 'flex';
        console.log('✨ 创建新对话');
    }

    toggleSidebar() {
        this.sidebar.classList.toggle('open');
    }

    scrollToBottom() {
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }

    showNotification(message, type) {
        type = type || 'info';
        
        const colors = {
            success: '#19c37d',
            error: '#ef4444',
            warning: '#f59e0b',
            info: '#3b82f6'
        };
        
        const notification = document.createElement('div');
        notification.style.cssText = 
            'position: fixed;' +
            'top: 20px;' +
            'right: 20px;' +
            'background: ' + colors[type] + ';' +
            'color: white;' +
            'padding: 16px 24px;' +
            'border-radius: 8px;' +
            'box-shadow: 0 4px 12px rgba(0,0,0,0.3);' +
            'z-index: 2000;' +
            'font-size: 14px;';
        
        notification.textContent = message;
        document.body.appendChild(notification);
        
        setTimeout(function() {
            notification.remove();
        }, 3000);
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML.replace(/\n/g, '<br>');
    }
}

// 初始化应用
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 应用初始化...');
    new ChatApp();
    console.log('✅ 应用初始化完成');
});