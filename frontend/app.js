class App {
    constructor() {
        this.apiUrl = 'http://localhost:5000/api';
        this.selectedModel = '';
        this.chatHistory = [];
        this.currentChatId = null;
        
        this.initElements();
        this.initEventListeners();
        this.initDetector();
        this.loadModels();
        this.checkDevice();
        
        // ⭐ 确保初始状态
        this.ensureInitialView();
        
        console.log('✅ 应用初始化完成');
    }

    initElements() {
        // 聊天元素
        this.messageInput = document.getElementById('messageInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.messages = document.getElementById('messages');
        this.welcomeScreen = document.getElementById('welcomeScreen');
        this.modelSelect = document.getElementById('modelSelect');
        this.menuToggle = document.getElementById('menuToggle');
        this.sidebar = document.getElementById('sidebar');
        
        // 检测器元素
        this.detectorInput = document.getElementById('detectorInput');
        this.detectBtn = document.getElementById('detectBtn');
        this.clearBtn = document.getElementById('clearBtn');
        this.resultCard = document.getElementById('resultCard');
        
        // 视图元素
        this.chatView = document.getElementById('chatView');
        this.detectorView = document.getElementById('detectorView');
        this.chatNavBtn = document.getElementById('chatNavBtn');
        this.detectorNavBtn = document.getElementById('detectorNavBtn');
        
        console.log('✅ 元素初始化:', {
            chatView: !!this.chatView,
            detectorView: !!this.detectorView,
            chatNavBtn: !!this.chatNavBtn,
            detectorNavBtn: !!this.detectorNavBtn
        });
    }

    initEventListeners() {
        // 发送按钮
        if (this.sendBtn) {
            this.sendBtn.addEventListener('click', () => this.sendMessage());
        }
        
        // 模型选择
        if (this.modelSelect) {
            this.modelSelect.addEventListener('change', (e) => this.selectModel(e.target.value));
        }
        
        // 菜单切换
        if (this.menuToggle) {
            this.menuToggle.addEventListener('click', () => this.toggleSidebar());
        }
        
        // 输入框事件
        if (this.messageInput) {
            this.messageInput.addEventListener('input', () => {
                this.autoResize(this.messageInput);
                this.updateSendButton();
            });
            
            this.messageInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendMessage();
                }
            });
        }

        // 建议卡片点击
        document.querySelectorAll('.suggestion-card').forEach(card => {
            card.addEventListener('click', () => {
                const prompt = card.dataset.prompt;
                if (this.messageInput) {
                    this.messageInput.value = prompt;
                    this.autoResize(this.messageInput);
                    this.updateSendButton();
                    this.messageInput.focus();
                }
            });
        });
    }

    // ========== 初始化检测器 ==========
    initDetector() {
        console.log('🔧 初始化检测器...');
        
        // AI 对话按钮
        if (this.chatNavBtn) {
            this.chatNavBtn.addEventListener('click', (e) => {
                e.preventDefault();
                console.log('🖱️ 点击了 AI 对话按钮');
                this.switchToChat();
            });
            console.log('✅ AI 对话按钮已绑定');
        } else {
            console.error('❌ 找不到 chatNavBtn 元素');
        }
        
        // 安全检测按钮
        if (this.detectorNavBtn) {
            this.detectorNavBtn.addEventListener('click', (e) => {
                e.preventDefault();
                console.log('🖱️ 点击了安全检测按钮');
                this.switchToDetector();
            });
            console.log('✅ 安全检测按钮已绑定');
        } else {
            console.error('❌ 找不到 detectorNavBtn 元素');
        }
        
        // 字符计数
        if (this.detectorInput) {
            this.detectorInput.addEventListener('input', () => {
                const count = this.detectorInput.value.length;
                const charCountEl = document.getElementById('charCount');
                if (charCountEl) {
                    charCountEl.textContent = `${count} 字符`;
                }
            });
            
            // Ctrl+Enter 快捷键
            this.detectorInput.addEventListener('keydown', (e) => {
                if (e.ctrlKey && e.key === 'Enter') {
                    this.runDetection();
                }
            });
        }
        
        // 检测按钮
        if (this.detectBtn) {
            this.detectBtn.addEventListener('click', () => this.runDetection());
        }
        
        // 清空按钮
        if (this.clearBtn) {
            this.clearBtn.addEventListener('click', () => {
                if (this.detectorInput) {
                    this.detectorInput.value = '';
                }
                const charCountEl = document.getElementById('charCount');
                if (charCountEl) {
                    charCountEl.textContent = '0 字符';
                }
                if (this.resultCard) {
                    this.resultCard.style.display = 'none';
                }
            });
        }
        
        console.log('✅ 检测器初始化完成');
    }

    // ========== 确保初始视图正确 ==========
    ensureInitialView() {
        console.log('🔄 设置初始视图...');
        
        // 移除所有 active 类
        if (this.chatView) this.chatView.classList.remove('active');
        if (this.detectorView) this.detectorView.classList.remove('active');
        if (this.chatNavBtn) this.chatNavBtn.classList.remove('active');
        if (this.detectorNavBtn) this.detectorNavBtn.classList.remove('active');
        
        // 只显示聊天视图
        if (this.chatView) this.chatView.classList.add('active');
        if (this.chatNavBtn) this.chatNavBtn.classList.add('active');
        
        console.log('✅ 初始视图设置完成 - 显示聊天界面');
        console.log('视图状态:', {
            chatViewActive: this.chatView?.classList.contains('active'),
            detectorViewActive: this.detectorView?.classList.contains('active')
        });
    }

    // ========== 切换到聊天 ==========
    switchToChat() {
        console.log('🔄 切换到聊天视图');
        
        // 移除所有 active
        this.chatView.classList.remove('active');
        this.detectorView.classList.remove('active');
        this.chatNavBtn.classList.remove('active');
        this.detectorNavBtn.classList.remove('active');
        
        // 激活聊天
        this.chatView.classList.add('active');
        this.chatNavBtn.classList.add('active');
        
        console.log('✅ 切换完成 - 聊天视图已激活');
    }

    // ========== 切换到检测器 ==========
    switchToDetector() {
        console.log('🔄 切换到检测器视图');
        
        // 移除所有 active
        this.chatView.classList.remove('active');
        this.detectorView.classList.remove('active');
        this.chatNavBtn.classList.remove('active');
        this.detectorNavBtn.classList.remove('active');
        
        // 激活检测器
        this.detectorView.classList.add('active');
        this.detectorNavBtn.classList.add('active');
        
        console.log('✅ 切换完成 - 检测器视图已激活');
    }

    // ========== 执行检测 ==========
    async runDetection() {
        const text = this.detectorInput.value.trim();
        
        if (!text) {
            this.showToast('请输入要检测的文本', 'warning');
            return;
        }
        
        this.detectBtn.disabled = true;
        this.detectBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 检测中...';
        
        try {
            const startTime = performance.now();
            
            const response = await fetch(this.apiUrl + '/detect', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });
            
            const result = await response.json();
            const detectTime = ((performance.now() - startTime) / 1000).toFixed(3);
            
            if (response.ok) {
                this.displayDetectionResult(result, detectTime);
            } else {
                this.showToast(result.error || '检测失败', 'error');
            }
        } catch (error) {
            console.error('检测失败:', error);
            this.showToast('网络错误，请检查后端服务', 'error');
        } finally {
            this.detectBtn.disabled = false;
            this.detectBtn.innerHTML = '<i class="fas fa-search"></i> 开始检测';
        }
    }

    // ========== 显示检测结果 ==========
    displayDetectionResult(result, detectTime) {
        this.resultCard.style.display = 'block';
        
        // 检测时间
        document.getElementById('timeBadge').textContent = `检测耗时: ${detectTime}s`;
        
        // 风险评分
        const score = result.risk_score || 0;
        document.getElementById('scoreNum').textContent = Math.round(score);
        
        // 更新圆环
        const circle = document.getElementById('circleProgress');
        const circumference = 377;
        const offset = circumference - (score / 100) * circumference;
        circle.style.strokeDashoffset = offset;
        
        // 风险等级
        const badges = {
            safe: '✅ 安全',
            low: '⚡ 低风险',
            medium: '⚠️ 中等风险',
            high: '🚨 高风险',
            critical: '🔴 严重风险'
        };
        document.getElementById('riskBadge').textContent = badges[result.risk_level] || badges.safe;
        
        // 风险判定
        const verdict = result.is_jailbreak ? 
            '🚫 检测到安全威胁，建议拦截' : 
            '✅ 未检测到安全威胁';
        document.getElementById('riskVerdict').textContent = verdict;
        
        // 统计数量
        document.getElementById('threatCount').textContent = result.matched_features?.length || 0;
        document.getElementById('keywordCount').textContent = result.suspicious_keywords?.length || 0;
        
        // 显示详细信息
        this.showThreats(result.matched_features);
        this.showKeywords(result.suspicious_keywords);
        this.showSuggestions(result.recommendations);
        
        // 滚动到结果
        this.resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    // ========== 显示威胁特征 ==========
    showThreats(features) {
        const card = document.getElementById('threatCard');
        const content = document.getElementById('threatContent');
        
        if (!features || features.length === 0) {
            card.style.display = 'none';
            return;
        }
        
        card.style.display = 'block';
        content.innerHTML = features.map(f => `
            <div class="analysis-tag">
                <div class="tag-title">${this.escapeHtml(f.name || '未知特征')}</div>
                <div class="tag-desc">${this.escapeHtml(f.description || '无描述')}</div>
            </div>
        `).join('');
    }

    // ========== 显示可疑关键词 ==========
    showKeywords(keywords) {
        const card = document.getElementById('keywordCard');
        const content = document.getElementById('keywordContent');
        
        if (!keywords || keywords.length === 0) {
            card.style.display = 'none';
            return;
        }
        
        card.style.display = 'block';
        content.innerHTML = keywords.map(k => `
            <div class="analysis-tag">
                <div class="tag-title">"${this.escapeHtml(k.keyword || '')}"</div>
                <div class="tag-desc">${this.escapeHtml(k.name || '')}</div>
            </div>
        `).join('');
    }

    // ========== 显示安全建议 ==========
    showSuggestions(suggestions) {
        const card = document.getElementById('suggestionCard');
        const content = document.getElementById('suggestionContent');
        
        if (!suggestions || suggestions.length === 0) {
            card.style.display = 'none';
            return;
        }
        
        card.style.display = 'block';
        content.innerHTML = suggestions.map(s => `
            <div class="analysis-tag">
                <div class="tag-title">${this.escapeHtml(s)}</div>
            </div>
        `).join('');
    }

    // ========== 聊天功能 ==========
    async sendMessage() {
        const message = this.messageInput.value.trim();
        
        if (!message) {
            this.showToast('请输入消息', 'warning');
            return;
        }
        
        if (!this.selectedModel) {
            this.showToast('请先选择模型', 'warning');
            return;
        }

        // 隐藏欢迎屏幕
        if (this.welcomeScreen) {
            this.welcomeScreen.style.display = 'none';
        }

        // 添加用户消息
        this.addMessage('user', message);
        this.messageInput.value = '';
        this.autoResize(this.messageInput);
        this.updateSendButton();
        
        this.sendBtn.disabled = true;
        
        // 显示加载动画
        const loadingId = this.addMessage('assistant', '正在思考...', true);

        try {
            const response = await fetch(this.apiUrl + '/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model: this.selectedModel,
                    message: message,
                    history: this.chatHistory
                })
            });

            const data = await response.json();
            
            // 移除加载消息
            const loadingEl = document.getElementById(loadingId);
            if (loadingEl) loadingEl.remove();
            
            if (response.ok) {
                // 添加助手回复
                this.addMessage('assistant', data.response);
                
                // 更新历史
                this.chatHistory.push(
                    { role: 'user', content: message },
                    { role: 'assistant', content: data.response }
                );
            } else {
                this.showToast(data.error || '请求失败', 'error');
            }
        } catch (error) {
            console.error('发送消息失败:', error);
            const loadingEl = document.getElementById(loadingId);
            if (loadingEl) loadingEl.remove();
            this.showToast('网络错误，请检查后端服务', 'error');
        } finally {
            this.sendBtn.disabled = false;
        }
    }

    addMessage(role, content, isLoading = false) {
        const messageDiv = document.createElement('div');
        const messageId = 'msg-' + Date.now();
        messageDiv.id = messageId;
        messageDiv.className = `message ${role}-message`;
        
        const avatar = role === 'user' ? 
            '<i class="fas fa-user"></i>' : 
            '<i class="fas fa-robot"></i>';
        
        const loadingClass = isLoading ? 'loading' : '';
        
        messageDiv.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content ${loadingClass}">
                <div class="message-text">${this.escapeHtml(content)}</div>
            </div>
        `;
        
        this.messages.appendChild(messageDiv);
        this.scrollToBottom();
        
        return messageId;
    }

    // ========== 工具函数 ==========
    
    async loadModels() {
        try {
            const response = await fetch(this.apiUrl + '/models');
            const data = await response.json();
            
            this.modelSelect.innerHTML = '<option value="">请选择模型...</option>';
            
            data.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.id;
                option.textContent = model.name;
                this.modelSelect.appendChild(option);
            });
            
            console.log('✅ 加载了', data.models.length, '个模型');
        } catch (error) {
            console.error('加载模型失败:', error);
            this.showToast('无法连接到后端服务', 'error');
        }
    }

    async checkDevice() {
        try {
            const response = await fetch(this.apiUrl + '/device');
            const data = await response.json();
            
            const deviceStatusEl = document.getElementById('deviceStatus');
            if (deviceStatusEl) {
                deviceStatusEl.textContent = data.device;
            }
            console.log('✅ 设备:', data.device);
        } catch (error) {
            console.error('检测设备失败:', error);
        }
    }

    selectModel(modelId) {
        this.selectedModel = modelId;
        this.updateSendButton();
        
        if (modelId) {
            const option = this.modelSelect.options[this.modelSelect.selectedIndex];
            const modelStatusEl = document.getElementById('modelStatus');
            if (modelStatusEl) {
                modelStatusEl.textContent = option.text;
            }
            this.showToast('已选择模型：' + option.text, 'success');
        }
    }

    toggleSidebar() {
        this.sidebar.classList.toggle('open');
    }

    autoResize(textarea) {
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px';
    }

    updateSendButton() {
        const hasText = this.messageInput.value.trim().length > 0;
        const hasModel = this.selectedModel !== '';
        this.sendBtn.disabled = !(hasText && hasModel);
    }

    scrollToBottom() {
        this.messages.scrollTop = this.messages.scrollHeight;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML.replace(/\n/g, '<br>');
    }

    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        
        const icons = {
            success: '<i class="fas fa-check-circle"></i>',
            error: '<i class="fas fa-times-circle"></i>',
            warning: '<i class="fas fa-exclamation-triangle"></i>',
            info: '<i class="fas fa-info-circle"></i>'
        };
        
        toast.innerHTML = `
            ${icons[type]}
            <span>${message}</span>
        `;
        
        const container = document.getElementById('toastContainer');
        container.appendChild(toast);
        
        setTimeout(() => {
            toast.style.animation = 'slideOutRight 0.3s ease';
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }
}

// 初始化应用
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 页面加载完成，初始化应用...');
    new App();
});
