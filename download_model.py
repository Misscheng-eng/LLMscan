"""
下载 AI 模型到本地
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def download_model(model_name, save_dir):
    """
    下载模型到本地
    
    Args:
        model_name: Hugging Face 模型名称
        save_dir: 保存目录
    """
    print("=" * 80)
    print(f"📥 开始下载模型: {model_name}")
    print("=" * 80)
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # 下载 tokenizer
        print("\n1️⃣ 下载 Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        tokenizer.save_pretrained(save_dir)
        print("   ✅ Tokenizer 下载完成")
        
        # 下载模型
        print("\n2️⃣ 下载模型文件（这可能需要几分钟到几十分钟）...")
        print("   💡 提示：首次下载会较慢，请耐心等待")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype="auto",
            low_cpu_mem_usage=True
        )
        model.save_pretrained(save_dir)
        print("   ✅ 模型下载完成")
        
        print("\n" + "=" * 80)
        print(f"✅ 模型已保存到: {save_dir}")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        import traceback
        traceback.print_exc()

# ============================================================
# 可下载的模型列表（按大小排序）
# ============================================================

MODELS = {
    "1": {
        "name": "Qwen2.5-1.5B",
        "hf_name": "Qwen/Qwen2.5-1.5B",
        "size": "~3 GB",
        "save_dir": "./models/Qwen2.5-1.5B",
        "description": "最小的模型，适合CPU运行"
    },
    "2": {
        "name": "Llama-3.2-1B",
        "hf_name": "meta-llama/Llama-3.2-1B",
        "size": "~2.5 GB",
        "save_dir": "./models/Llama-3.2-1B",
        "description": "Meta最小模型"
    },
    "3": {
        "name": "Qwen2.5-7B",
        "hf_name": "Qwen/Qwen2.5-7B",
        "size": "~15 GB",
        "save_dir": "./models/Qwen2.5-7B",
        "description": "中型模型，需要较好的硬件"
    },
    "4": {
        "name": "Llama-3.1-8B",
        "hf_name": "meta-llama/Llama-3.1-8B",
        "size": "~16 GB",
        "save_dir": "./models/Llama-3.1-8B",
        "description": "大型模型，需要16GB+ RAM"
    }
}

if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("🤖 AI 模型下载工具")
    print("=" * 80)
    print("\n可下载的模型：\n")
    
    for key, model in MODELS.items():
        print(f"{key}. {model['name']}")
        print(f"   大小: {model['size']}")
        print(f"   说明: {model['description']}")
        print()
    
    choice = input("请选择要下载的模型编号 (1-4): ").strip()
    
    if choice in MODELS:
        model_info = MODELS[choice]
        print(f"\n✅ 已选择: {model_info['name']}")
        print(f"📦 大小: {model_info['size']}")
        print(f"💾 保存路径: {model_info['save_dir']}")
        
        confirm = input("\n确认下载？(y/n): ").strip().lower()
        
        if confirm == 'y':
            download_model(model_info['hf_name'], model_info['save_dir'])
        else:
            print("❌ 已取消下载")
    else:
        print("❌ 无效的选择")