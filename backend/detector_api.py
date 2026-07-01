"""
Jailbreak/Toxic Detection API
基于 hidden_state_detector.py 实现的检测服务
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
from joblib import load
import json
import time
from datetime import datetime

# 导入您的检测模块
from public_func.hidden_state_detector import (
    extract_hidden_states_batched,
    load_feature_cache
)
from utils.modelUtils import ModelAndTokenizer
from utils.utils import prepare_prompt

app = Flask(__name__)
CORS(app)

# ============================================================
# 全局配置
# ============================================================

CONFIG = {
    "models": [
        {
            "id": "llama2-7b",
            "name": "LLaMA 2-7B",
            "path": "meta-llama/Llama-2-7b-hf",
            "description": "Meta的LLaMA 2模型,适合快速检测",
            "classifier_path": "models/classifiers/llama2_7b_detector.joblib"
        },
        {
            "id": "llama3-8b",
            "name": "LLaMA 3.1-8B",
            "path": "meta-llama/Meta-Llama-3.1-8B",
            "description": "最新的LLaMA 3.1模型,检测精度更高",
            "classifier_path": "models/classifiers/llama3_8b_detector.joblib"
        }
    ],
    "n_last_layers": 5,
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "batch_size": 8,
    "detection_types": ["jailbreak", "toxic", "safe"]
}

# 缓存已加载的模型和分类器
loaded_models = {}
loaded_classifiers = {}


# ============================================================
# 模型管理
# ============================================================

def get_model_and_tokenizer(model_id):
    """获取或加载模型"""
    if model_id in loaded_models:
        return loaded_models[model_id]
    
    model_info = next((m for m in CONFIG["models"] if m["id"] == model_id), None)
    if not model_info:
        raise ValueError(f"未知模型ID: {model_id}")
    
    print(f"加载模型: {model_info['name']}...")
    mt = ModelAndTokenizer(
        model_info["path"],
        low_cpu_mem_usage=True,
        use_cache=False,
        device=CONFIG["device"]
    )
    loaded_models[model_id] = mt
    return mt


def get_classifier(model_id):
    """获取或加载分类器"""
    if model_id in loaded_classifiers:
        return loaded_classifiers[model_id]
    
    model_info = next((m for m in CONFIG["models"] if m["id"] == model_id), None)
    if not model_info:
        raise ValueError(f"未知模型ID: {model_id}")
    
    classifier_path = model_info["classifier_path"]
    if not os.path.exists(classifier_path):
        raise FileNotFoundError(f"分类器文件不存在: {classifier_path}")
    
    print(f"加载分类器: {classifier_path}")
    clf = load(classifier_path)
    loaded_classifiers[model_id] = clf
    return clf


# ============================================================
# API端点
# ============================================================

@app.route('/api/models', methods=['GET'])
def get_available_models():
    """获取可用模型列表"""
    return jsonify({
        "models": CONFIG["models"],
        "device": CONFIG["device"],
        "cuda_available": torch.cuda.is_available()
    })


@app.route('/api/detect', methods=['POST'])
def detect_prompt():
    """
    检测单个prompt是否为jailbreak/toxic
    
    请求体:
    {
        "model": "llama2-7b",
        "text": "用户输入的prompt",
        "return_features": false  // 可选,是否返回特征向量
    }
    """
    try:
        data = request.json
        model_id = data.get('model')
        text = data.get('text', '').strip()
        return_features = data.get('return_features', False)
        
        if not model_id or not text:
            return jsonify({"error": "缺少model或text参数"}), 400
        
        # 准备prompt
        prompt = prepare_prompt(text)
        
        # 获取模型和分类器
        mt = get_model_and_tokenizer(model_id)
        clf = get_classifier(model_id)
        
        # 提取hidden states特征
        start_time = time.time()
        features = extract_hidden_states_batched(
            [prompt], 
            mt, 
            n_last_layers=CONFIG["n_last_layers"],
            batch_size=1,
            device=CONFIG["device"]
        )
        extraction_time = time.time() - start_time
        
        # 预测
        prediction = clf.predict(features)[0]
        probability = clf.predict_proba(features)[0]
        
        # 构建响应
        result = {
            "prediction": "jailbreak" if prediction == 1 else "safe",
            "confidence": float(max(probability)),
            "probabilities": {
                "safe": float(probability[0]),
                "jailbreak": float(probability[1])
            },
            "model": model_id,
            "processing_time": round(extraction_time, 3),
            "timestamp": datetime.now().isoformat()
        }
        
        if return_features:
            result["features"] = features.tolist()
        
        return jsonify(result)
        
    except Exception as e:
        print(f"检测错误: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/detect/batch', methods=['POST'])
def detect_batch():
    """
    批量检测多个prompts
    
    请求体:
    {
        "model": "llama2-7b",
        "texts": ["prompt1", "prompt2", ...],
        "max_batch_size": 16  // 可选
    }
    """
    try:
        data = request.json
        model_id = data.get('model')
        texts = data.get('texts', [])
        max_batch = data.get('max_batch_size', CONFIG["batch_size"])
        
        if not model_id or not texts:
            return jsonify({"error": "缺少model或texts参数"}), 400
        
        # 准备prompts
        prompts = [prepare_prompt(t) for t in texts]
        
        # 获取模型和分类器
        mt = get_model_and_tokenizer(model_id)
        clf = get_classifier(model_id)
        
        # 提取特征
        start_time = time.time()
        features = extract_hidden_states_batched(
            prompts, 
            mt, 
            n_last_layers=CONFIG["n_last_layers"],
            batch_size=max_batch,
            device=CONFIG["device"]
        )
        
        # 批量预测
        predictions = clf.predict(features)
        probabilities = clf.predict_proba(features)
        
        # 构建结果
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                "index": i,
                "text": texts[i][:100] + "..." if len(texts[i]) > 100 else texts[i],
                "prediction": "jailbreak" if pred == 1 else "safe",
                "confidence": float(max(prob)),
                "probabilities": {
                    "safe": float(prob[0]),
                    "jailbreak": float(prob[1])
                }
            })
        
        return jsonify({
            "results": results,
            "total": len(texts),
            "jailbreak_count": int(sum(predictions)),
            "safe_count": int(len(predictions) - sum(predictions)),
            "processing_time": round(time.time() - start_time, 3),
            "model": model_id
        })
        
    except Exception as e:
        print(f"批量检测错误: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/analyze', methods=['POST'])
def analyze_prompt():
    """
    深度分析prompt,返回详细信息
    
    请求体:
    {
        "model": "llama2-7b",
        "text": "用户输入的prompt",
        "include_visualization": false  // 是否生成可视化数据
    }
    """
    try:
        data = request.json
        model_id = data.get('model')
        text = data.get('text', '').strip()
        include_viz = data.get('include_visualization', False)
        
        if not model_id or not text:
            return jsonify({"error": "缺少model或text参数"}), 400
        
        # 检测
        detect_result = detect_prompt()
        if detect_result[1] != 200:  # 如果检测失败
            return detect_result
        
        result = detect_result[0].get_json()
        
        # 添加分析信息
        result["analysis"] = {
            "text_length": len(text),
            "word_count": len(text.split()),
            "risk_level": "高" if result["probabilities"]["jailbreak"] > 0.8 
                         else "中" if result["probabilities"]["jailbreak"] > 0.5 
                         else "低",
            "recommendation": get_recommendation(result["probabilities"]["jailbreak"])
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"分析错误: {str(e)}")
        return jsonify({"error": str(e)}), 500


def get_recommendation(jailbreak_prob):
    """根据概率给出建议"""
    if jailbreak_prob > 0.8:
        return "强烈建议拒绝该输入,存在高风险越狱尝试"
    elif jailbreak_prob > 0.5:
        return "建议对该输入进行人工审核"
    else:
        return "该输入相对安全,可以正常处理"


@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        "status": "ok",
        "loaded_models": list(loaded_models.keys()),
        "loaded_classifiers": list(loaded_classifiers.keys()),
        "device": CONFIG["device"],
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """获取系统统计信息"""
    return jsonify({
        "total_models": len(CONFIG["models"]),
        "loaded_models": len(loaded_models),
        "device": CONFIG["device"],
        "cuda_available": torch.cuda.is_available(),
        "n_last_layers": CONFIG["n_last_layers"]
    })


# ============================================================
# 启动服务
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Jailbreak/Toxic Detection API Server")
    print("=" * 60)
    print(f"Device: {CONFIG['device']}")
    print(f"Available models: {len(CONFIG['models'])}")
    print("=" * 60)
    
    app.run(
        debug=True, 
        host='0.0.0.0', 
        port=5000,
        threaded=True  # 支持并发请求
    )