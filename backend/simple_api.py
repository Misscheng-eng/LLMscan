"""
简化版 AI 模型交互 API
用户可以选择模型、输入prompt、获取响应
支持越狱检测功能
"""

import sys
import os

# 确保能导入项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import time
from datetime import datetime

# 强制输出到stdout（Windows环境特别需要）
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

# 禁用输出缓冲
os.environ['PYTHONUNBUFFERED'] = '1'

print("=" * 80)
print("开始导入模块...")
print("=" * 80)

# 导入工具模块
try:
    from utils.modelUtils import ModelAndTokenizer
    from utils.utils import prepare_prompt
    print("✅ 成功导入工具模块")
except ImportError as e:
    print(f"⚠️  导入工具模块失败: {e}")
    print("提示: 部分功能可能受限")

# 导入越狱检测器
try:
    from simple_jailbreak_detector import SimpleJailbreakDetector
    print("✅ 成功导入越狱检测器")
    JAILBREAK_DETECTION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  导入越狱检测器失败: {e}")
    print("提示: 越狱检测功能将被禁用")
    JAILBREAK_DETECTION_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# 初始化越狱检测器
JAILBREAK_DETECTION_ENABLED = False
jailbreak_detector = None

if JAILBREAK_DETECTION_AVAILABLE:
    try:
        print("🔧 初始化越狱检测器...")
        jailbreak_detector = SimpleJailbreakDetector()
        JAILBREAK_DETECTION_ENABLED = True
        print("✅ 越狱检测器已启用")
    except Exception as e:
        print(f"⚠️  越狱检测器初始化失败: {e}")
        print("提示: 将继续运行但不进行越狱检测")

# 统计数据
detection_stats = {
    "total_requests": 0,
    "blocked_requests": 0,
    "risk_levels": {
        "safe": 0,
        "low": 0,
        "medium": 0,
        "high": 0,
        "critical": 0
    }
}

# ============================================================
# 配置 - 使用您的实际模型
# ============================================================

AVAILABLE_MODELS = [
    {
        "id": "qwen2.5-1.5b",
        "name": "Qwen2.5-1.5B",
        "path": "Qwen/Qwen2.5-1.5B",
        "description": "轻量级Qwen模型，1.5B参数，速度快，适合快速测试",
        "icon": "⚡"
    },
    {
        "id": "llama3.2-1b",
        "name": "Llama-3.2-1B",
        "path": "meta-llama/Llama-3.2-1B",
        "description": "Meta最新的小型模型，1B参数，高效快速",
        "icon": "🦙"
    },
    {
        "id": "qwen2.5-7b",
        "name": "Qwen2.5-7B",
        "path": "Qwen/Qwen2.5-7B",
        "description": "Qwen中型模型，7B参数，性能强劲",
        "icon": "🚀"
    },
    {
        "id": "llama3.1-8b",
        "name": "Llama-3.1-8B",
        "path": "meta-llama/Llama-3.1-8B",
        "description": "Meta LLaMA 3.1模型，8B参数，性能卓越",
        "icon": "🦙"
    }
]

# 缓存已加载的模型
loaded_models = {}
device = "cuda:0" if torch.cuda.is_available() else "cpu"

print("\n" + "=" * 80)
print("🚀 AI模型交互服务启动中...")
print("=" * 80)
print(f"📍 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"💻 设备: {device}")
print(f"🔥 CUDA可用: {'是' if torch.cuda.is_available() else '否'}")
print(f"🐍 Python版本: {sys.version.split()[0]}")
print(f"🔥 PyTorch版本: {torch.__version__}")

if torch.cuda.is_available():
    print(f"🎮 GPU型号: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("⚠️  警告: 运行在CPU模式，推理速度会较慢")

print(f"📦 可用模型数: {len(AVAILABLE_MODELS)}")
for model in AVAILABLE_MODELS:
    print(f"   {model['icon']} {model['name']} ({model['id']})")

print(f"🛡️ 越狱检测: {'已启用' if JAILBREAK_DETECTION_ENABLED else '已禁用'}")
print("=" * 80 + "\n")


# ============================================================
# 模型管理
# ============================================================

def get_or_load_model(model_id):
    """获取或加载模型"""
    
    if model_id in loaded_models:
        print(f"✅ 使用已缓存的模型: {model_id}")
        return loaded_models[model_id]
    
    model_config = next((m for m in AVAILABLE_MODELS if m["id"] == model_id), None)
    if not model_config:
        raise ValueError(f"未知模型: {model_id}")
    
    print("\n" + "=" * 80)
    print(f"⏳ 开始加载模型: {model_config['name']}")
    print("=" * 80)
    print(f"📂 模型路径: {model_config['path']}")
    print(f"💻 设备: {device}")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import gc
        
        # 加载 tokenizer
        print("\n📝 步骤 1/2: 加载 Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_config["path"],
            trust_remote_code=True,
            use_fast=True
        )
        print("   ✅ Tokenizer 加载完成")
        
        # 加载模型
        print("\n🤖 步骤 2/2: 加载模型权重...")
        
        # 清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        model = AutoModelForCausalLM.from_pretrained(
            model_config["path"],
            trust_remote_code=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map="cpu",
            offload_folder="./offload",
            offload_state_dict=True,
        )
        
        model.eval()
        print("   ✅ 模型加载完成")
        
        # 创建包装对象
        class ModelWrapper:
            def __init__(self, model, tokenizer, model_id):
                self.model = model
                self.tokenizer = tokenizer
                self.model_id = model_id
                self.num_layers = getattr(model.config, 'num_hidden_layers', 0)
        
        mt = ModelWrapper(model, tokenizer, model_id)
        loaded_models[model_id] = mt
        
        load_time = time.time() - start_time
        
        print("\n" + "=" * 80)
        print("✅ 模型加载成功！")
        print("=" * 80)
        print(f"⏱️  总耗时: {load_time:.2f} 秒")
        print(f"📊 模型层数: {mt.num_layers}")
        print("=" * 80)
        print()
        
        return mt
        
    except Exception as e:
        print(f"\n❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        raise


# ============================================================
# API 端点
# ============================================================

@app.route('/api/models', methods=['GET'])
def get_models():
    """获取可用模型列表"""
    print(f"\n📋 [GET /api/models] 收到请求")
    
    response_data = {
        "models": AVAILABLE_MODELS,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "loaded_models": list(loaded_models.keys()),
        "jailbreak_detection_enabled": JAILBREAK_DETECTION_ENABLED
    }
    
    print(f"✅ 返回 {len(AVAILABLE_MODELS)} 个可用模型")
    return jsonify(response_data)


@app.route('/api/device', methods=['GET'])
def get_device():
    """获取设备信息"""
    print(f"\n💻 [GET /api/device] 收到请求")
    
    device_info = {
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "status": "ok"
    }
    
    if torch.cuda.is_available():
        device_info["gpu_name"] = torch.cuda.get_device_name(0)
        device_info["gpu_memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    
    print(f"✅ 返回设备信息: {device}")
    return jsonify(device_info)


@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def chat():
    """处理对话请求"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.json
        model_id = data.get('model')
        message = data.get('message', '').strip()
        history = data.get('history', [])
        
        print("\n" + "=" * 80)
        print("💬 [POST /api/chat] 收到对话请求")
        print("=" * 80)
        print(f"🤖 模型ID: {model_id}")
        print(f"📝 用户消息: {message[:100]}...")
        print(f"📚 历史条数: {len(history)} 条")
        print(f"⏰ 时间: {datetime.now().strftime('%H:%M:%S')}")
        
        if not model_id or not message:
            return jsonify({"error": "缺少必要参数"}), 400
        
        # 越狱检测
        if JAILBREAK_DETECTION_ENABLED and jailbreak_detector:
            print("\n🔍 开始越狱检测...")
            detection_start = time.time()
            
            try:
                detection_result = jailbreak_detector.detect(message)
                detection_time = time.time() - detection_start
                
                # 验证返回数据
                if not isinstance(detection_result, dict):
                    raise ValueError(f"检测器返回类型错误: {type(detection_result)}")
                
                required_keys = ['is_jailbreak', 'risk_score', 'risk_level']
                missing_keys = [key for key in required_keys if key not in detection_result]
                if missing_keys:
                    raise ValueError(f"检测结果缺少键: {missing_keys}")
                
                risk_emoji = {
                    "safe": "✅",
                    "low": "⚡",
                    "medium": "⚠️",
                    "high": "🚨",
                    "critical": "🔴"
                }
                
                risk_level = detection_result.get('risk_level', 'medium').lower()
                risk_score = detection_result.get('risk_score', 0)
                is_jailbreak = detection_result.get('is_jailbreak', False)
                
                print(f"⏱️  检测耗时: {detection_time:.3f}秒")
                print(f"{risk_emoji.get(risk_level, '⚠️')} 风险等级: {risk_level.upper()}")
                print(f"📊 风险评分: {risk_score}/100")
                print(f"🛡️ 是否拦截: {'是' if is_jailbreak else '否'}")
                
                if detection_result.get('matched_features'):
                    print(f"   检测到 {len(detection_result['matched_features'])} 个越狱特征")
                    for feature in detection_result['matched_features'][:3]:
                        print(f"   • {feature.get('name', '未知特征')}")
                
                # 更新统计
                detection_stats["total_requests"] += 1
                detection_stats["risk_levels"][risk_level] += 1
                
                if is_jailbreak:
                    detection_stats["blocked_requests"] += 1
                    print(f"🚫 请求被拦截")
                    
                    # 构建拒绝消息
                    reject_message = "⚠️ 安全检测：检测到潜在的不当请求\n\n"
                    reject_message += f"📊 风险评分：{risk_score}/100\n"
                    reject_message += f"🎯 风险等级：{risk_level.upper()}\n\n"
                    
                    if detection_result.get('matched_features'):
                        reject_message += "检测到的问题：\n"
                        for i, feature in enumerate(detection_result['matched_features'][:3], 1):
                            reject_message += f"{i}. {feature.get('name', '未知')}\n"
                            reject_message += f"   {feature.get('description', '无描述')}\n"
                    
                    if detection_result.get('recommendations'):
                        reject_message += "\n💡 建议：\n"
                        for rec in detection_result['recommendations'][:3]:
                            reject_message += f"• {rec}\n"
                    
                    reject_message += "\n我是一个遵循伦理准则的AI助手，无法响应可能违反使用规范的指令。"
                    
                    return jsonify({
                        "response": reject_message,
                        "model": model_id,
                        "generation_time": detection_time,
                        "tokens_generated": 0,
                        "tokens_per_second": 0,
                        "jailbreak_detected": True,
                        "detection_result": {
                            "risk_score": risk_score,
                            "risk_level": risk_level,
                            "matched_features": detection_result.get('matched_features', []),
                            "suspicious_keywords": detection_result.get('suspicious_keywords', []),
                            "recommendations": detection_result.get('recommendations', [])
                        }
                    })
                
            except Exception as e:
                print(f"⚠️  越狱检测失败: {e}")
                print("   继续处理请求...")
                import traceback
                traceback.print_exc()
                detection_stats["total_requests"] += 1
        
        # 构建对话上下文
        conversation = ""
        for msg in history[-6:]:
            role = msg.get('role')
            content = msg.get('content')
            if role == 'user':
                conversation += f"User: {content}\n"
            elif role == 'assistant':
                conversation += f"Assistant: {content}\n"
        
        conversation += f"User: {message}\nAssistant:"
        
        print(f"📝 完整上下文长度: {len(conversation)} 字符")
        print("⏳ 开始生成回复...")
        
        # 加载模型
        mt = get_or_load_model(model_id)
        
        # 生成回复
        start_time = time.time()
        
        print("🔤 开始Tokenize...")
        inputs = mt.tokenizer(
            conversation,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        
        if hasattr(inputs, 'to') and callable(inputs.to):
            inputs = inputs.to(device)
        
        input_length = inputs.input_ids.shape[1]
        print(f"✅ Tokenize完成，输入tokens数: {input_length}")
        
        print("🎯 开始Generate...")
        with torch.no_grad():
            outputs = mt.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=mt.tokenizer.pad_token_id,
                eos_token_id=mt.tokenizer.eos_token_id
            )
        
        print("🔤 开始Decode...")
        generated_text = mt.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取回复
        if "Assistant:" in generated_text:
            response_text = generated_text.split("Assistant:")[-1].strip()
        else:
            response_text = generated_text[len(conversation):].strip()
        
        if "User:" in response_text:
            response_text = response_text.split("User:")[0].strip()
        
        gen_time = time.time() - start_time
        tokens_gen = len(outputs[0]) - input_length
        tokens_per_sec = tokens_gen / gen_time if gen_time > 0 else 0
        
        print("\n" + "=" * 60)
        print("✅ 生成完成!")
        print("=" * 60)
        print(f"⏱️  耗时: {gen_time:.2f} 秒")
        print(f"📊 输入tokens: {input_length}")
        print(f"📊 生成tokens: {tokens_gen}")
        print(f"⚡ 速度: {tokens_per_sec:.2f} tokens/秒")
        print(f"📤 回复: {response_text[:100]}...")
        print("=" * 60)
        print()
        
        return jsonify({
            "response": response_text,
            "model": model_id,
            "generation_time": round(gen_time, 2),
            "tokens_generated": tokens_gen,
            "tokens_per_second": round(tokens_per_sec, 2),
            "jailbreak_detected": False
        })
        
    except Exception as e:
        print(f"\n❌ 对话处理错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/detect', methods=['POST', 'OPTIONS'])
def detect_jailbreak():
    """独立的越狱检测接口"""
    
    # 处理 CORS 预检请求
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response, 200
    
    # 检查检测器是否可用
    if not JAILBREAK_DETECTION_ENABLED or not jailbreak_detector:
        print("❌ 越狱检测器未启用")
        return jsonify({
            "error": "越狱检测功能未启用",
            "message": "检测器未正确初始化"
        }), 503
    
    try:
        # 获取请求数据
        data = request.get_json()
        
        if not data:
            print("❌ 未收到 JSON 数据")
            return jsonify({"error": "无效的 JSON 数据"}), 400
        
        text = data.get('text', '').strip()
        
        if not text:
            print("❌ 文本为空")
            return jsonify({"error": "文本不能为空"}), 400
        
        print("\n" + "=" * 80)
        print("🔍 [POST /api/detect] 越狱检测请求")
        print("=" * 80)
        print(f"📝 文本: {text[:100]}...")
        print(f"📏 长度: {len(text)} 字符")
        print(f"⏰ 时间: {datetime.now().strftime('%H:%M:%S')}")
        
        # 执行检测
        detection_start = time.time()
        result = jailbreak_detector.detect(text)
        detection_time = time.time() - detection_start
        
        # 验证结果
        if not isinstance(result, dict):
            raise ValueError(f"检测器返回类型错误: {type(result)}")
        
        required_keys = ['is_jailbreak', 'risk_score', 'risk_level']
        missing_keys = [key for key in required_keys if key not in result]
        if missing_keys:
            raise ValueError(f"检测结果缺少键: {missing_keys}")
        
        # 更新统计
        detection_stats["total_requests"] += 1
        risk_level = result.get('risk_level', 'medium').lower()
        detection_stats["risk_levels"][risk_level] += 1
        if result.get('is_jailbreak', False):
            detection_stats["blocked_requests"] += 1
        
        # 打印结果
        risk_emoji = {
            "safe": "✅",
            "low": "⚡",
            "medium": "⚠️",
            "high": "🚨",
            "critical": "🔴"
        }
        
        print(f"\n⏱️  检测耗时: {detection_time:.3f}秒")
        print(f"{risk_emoji.get(risk_level, '⚠️')} 风险等级: {risk_level.upper()}")
        print(f"📊 风险评分: {result['risk_score']}/100")
        print(f"🛡️ 是否越狱: {'是' if result['is_jailbreak'] else '否'}")
        
        if result.get('matched_features'):
            print(f"🔍 匹配特征: {len(result['matched_features'])} 个")
            for feature in result['matched_features'][:3]:
                print(f"   • {feature.get('name', '未知')}")
        
        if result.get('suspicious_keywords'):
            print(f"🔑 可疑关键词: {len(result['suspicious_keywords'])} 个")
        
        print("=" * 80)
        
        # 返回结果
        response = jsonify(result)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 200
        
    except Exception as e:
        print(f"\n❌ 检测失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": f"服务器错误: {str(e)}",
            "type": type(e).__name__
        }), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """获取检测统计"""
    print(f"\n📊 [GET /api/stats] 收到请求")
    
    total = detection_stats["total_requests"]
    blocked = detection_stats["blocked_requests"]
    block_rate = round((blocked / total * 100) if total > 0 else 0, 1)
    
    stats_data = {
        "total_requests": total,
        "blocked_requests": blocked,
        "block_rate": block_rate,
        "risk_distribution": detection_stats["risk_levels"],
        "detector_enabled": JAILBREAK_DETECTION_ENABLED
    }
    
    print(f"✅ 总请求: {total}, 拦截: {blocked}, 拦截率: {block_rate}%")
    return jsonify(stats_data)


@app.route('/api/health', methods=['GET'])
def health():
    """健康检查"""
    print(f"\n🏥 [GET /api/health] 健康检查")
    
    health_data = {
        "status": "ok",
        "device": device,
        "loaded_models": list(loaded_models.keys()),
        "cuda_available": torch.cuda.is_available(),
        "jailbreak_detection_enabled": JAILBREAK_DETECTION_ENABLED,
        "timestamp": datetime.now().isoformat()
    }
    
    if torch.cuda.is_available():
        health_data["gpu_name"] = torch.cuda.get_device_name(0)
        health_data["gpu_memory_allocated"] = f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB"
        health_data["gpu_memory_reserved"] = f"{torch.cuda.memory_reserved(0) / 1024**3:.2f} GB"
    
    print(f"✅ 系统状态正常，已加载模型: {len(loaded_models)} 个")
    return jsonify(health_data)


@app.route('/api/test', methods=['GET'])
def test():
    """测试端点"""
    print("🧪 [GET /api/test] 测试请求")
    return jsonify({
        "status": "ok",
        "message": "后端服务运行正常",
        "timestamp": datetime.now().isoformat(),
        "detector_available": JAILBREAK_DETECTION_ENABLED
    })


# ============================================================
# 启动服务
# ============================================================

if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("🎯 服务器准备就绪，等待连接...")
    print("=" * 80)
    print("🌐 访问地址:")
    print("   - 本地: http://localhost:5000")
    print("   - 网络: http://0.0.0.0:5000")
    print("\n📖 API端点:")
    print("   GET  /api/test     - 测试连接")
    print("   GET  /api/models   - 获取模型列表")
    print("   GET  /api/device   - 获取设备信息")
    print("   POST /api/chat     - 发送对话消息")
    print("   POST /api/detect   - 越狱检测（独立）")
    print("   GET  /api/stats    - 获取统计信息")
    print("   GET  /api/health   - 健康检查")
    print("\n🛡️  越狱检测: " + ("✅ 已启用" if JAILBREAK_DETECTION_ENABLED else "❌ 未启用"))
    print("=" * 80)
    print("🔥 服务器启动中...")
    print("=" * 80 + "\n")
    
    # 启动Flask应用
    app.run(
        debug=False,
        host='0.0.0.0',
        port=5000,
        threaded=True,
        use_reloader=False
    )
