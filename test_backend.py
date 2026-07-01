"""
后端测试脚本 - 验证输出是否正常
"""

import sys
print("="*60, flush=True)
print("🧪 后端测试脚本启动", flush=True)
print("="*60, flush=True)
print(f"Python版本: {sys.version}", flush=True)
print(f"标准输出: {sys.stdout}", flush=True)
print(f"标准错误: {sys.stderr}", flush=True)

# 测试torch
try:
    import torch
    print(f"✅ Torch版本: {torch.__version__}", flush=True)
    print(f"✅ CUDA可用: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}", flush=True)
except Exception as e:
    print(f"❌ Torch导入失败: {e}", flush=True)

# 测试Flask
try:
    from flask import Flask
    app = Flask(__name__)
    
    @app.route('/test')
    def test():
        print("收到测试请求", flush=True)
        return "OK"
    
    print("✅ Flask导入成功", flush=True)
    print("启动测试服务器...", flush=True)
    app.run(host='0.0.0.0', port=5001, debug=False)
    
except Exception as e:
    print(f"❌ Flask测试失败: {e}", flush=True)