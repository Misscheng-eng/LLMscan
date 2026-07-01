# LLMScan 项目进展文档

> 最后更新：2026-06-16

---

## 一、当前状态

**项目已成功运行到因果分析阶段，模型推理正常，待完成检测器训练。**

---

## 二、环境配置变更

| 组件 | 原始版本 | 当前版本 | 变更原因 |
|------|---------|---------|---------|
| transformers | 4.33.3 | 4.46.3 | 支持 Qwen2.5 的 Qwen2Tokenizer |
| accelerate | 0.21.0 | 1.0.1 | Qwen 加载需要 `accelerate>=0.26.0` |
| torch | 2.4.1+cpu | 2.4.1+cu124 | 启用 GPU 推理 (RTX 4060 8GB) |
| huggingface-hub | 0.16.4 | 0.36.2 | 随 transformers 升级 |

### 环境变量
```powershell
$env:HF_HOME = "D:\huggingface_cache\huggingface"   # 模型缓存移至 D 盘
$env:HF_ENDPOINT = "https://hf-mirror.com"           # 国内镜像加速下载
```

---

## 三、代码修改清单

### 3.1 依赖导入修复

#### import 路径修复 (9 个文件)
**问题**：`sys.path.append(os.getcwd())` 依赖当前工作目录，离开项目目录运行时报 `ModuleNotFoundError`

**修改**：改为基于 `__file__` 的绝对路径计算
- `public_func/` 下 5 个文件：`sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))`
- 根目录 3 个文件：`sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))`
- `lllm/` 下 1 个文件：同 public_func

#### openai 模块懒加载 (12 个文件)
**问题**：`import openai` 触发 aiohttp SSL 错误 `[ASN1: NOT_ENOUGH_DATA]`（Windows 证书问题）

**修改**：
- 所有模块级 `import openai` 注释掉（12 个文件）
- `lllm/utils.py`、`lllm/dialogue_classes.py`、`data/convert_facts_to_questions.py` 改为函数内懒加载——仅当实际调用 OpenAI API 时才触发
- 相应 `openai.api_key = ...` 行注释掉

#### bias_detection 模块懒加载 (7 个文件)
**问题**：`bias_detection.TrustGPT` 是外部依赖（需单独安装 TrustGPT），模块级导入阻塞所有任务

**修改**：
- 模块级 `from bias_detection.TrustGPT...` 注释掉（7 个文件）
- `analyse_causality_toxic()` 函数内添加懒加载——仅当 `task='toxic'` 时才触发

#### 泄露 API Key 处理
- `lllm/dialogue_classes.py` 第 586 行硬编码的 OpenAI API Key 已注释

### 3.2 模型兼容性修复

#### 新增 Qwen2.5 模型支持 (4 个文件)
**问题**：`_can_answer` 列的模型名映射只认识 Llama/Mistral，Qwen 走到 else 分支时 `alternative_model_name` 未定义

**修改**：在以下文件中添加 Qwen 映射
```python
if "Qwen" in model_name:
    alternative_model_name = "llama-7b"
```
修改文件：`causality_analysis.py`、`causality_analysis_combine.py`、`causality_analysis_prompt.py`、`causality_distribution_map.py`

### 3.3 配置与入口修复

#### 数据集硬编码修复
**问题**：主入口 `causality_analysis.py` 第 1942 行硬编码 `dataset = MathematicalProblems()`，忽略 `parameters.json` 配置

**修改**：改为注释掉，使用参数配置的数据集

#### train_detector 变量缺失修复
**问题**：`train_detector()` else 分支调用 `get_aie_kurt` 时引用未定义的 `dataset_name_to_object`

**修改**：在函数开头添加 `dataset_name_to_object = {}`

### 3.4 运行配置 (`parameters.json`)

```json
{
    "model_path": "Qwen/",
    "model_name": "Qwen2.5-1.5B-Instruct",
    "task": "lie",
    "if_causality_analysis": true,
    "if_detect": true,
    "dataset": "Questions1000()",
    "target": "layer",
    "saving_dir": "outputs_lie/qwen2.5-1.5b/"
}
```

---

## 四、当前模型

| 属性 | 值 |
|------|-----|
| 模型 | Qwen/Qwen2.5-1.5B-Instruct |
| 大小 | 2.9GB（存储在 D 盘） |
| 推理设备 | NVIDIA RTX 4060 Laptop (8GB) |
| 推理框架 | PyTorch 2.4.1 + CUDA 12.4 |

---

## 五、已验证的 Pipeline 阶段

| 阶段 | 状态 | 说明 |
|------|------|------|
| 模型加载 | ✅ 通过 | Qwen2.5-1.5B 成功加载到 GPU |
| 文本生成 | ✅ 通过 | 已生成谎言回答示例："Spanish is not spoken in Argentina" |
| 因果分析 (Layer) | ✅ 通过 | 对所有层对做短路实验，生成 AIE 向量 |
| 数据持久化 | ✅ 通过 | AIE 结果保存到 `data/processed_questions/` |
| 检测器训练 | ⚠️ 待跑 | 已修复 bug，待下一次运行完成 |
| 检测器评估 | ⚠️ 待跑 | 训练完成后自动进行 |

---

## 六、输出文件位置

| 文件 | 说明 |
|------|------|
| `data/processed_questions/Questions1000.json` | 含 Qwen AIE 值的处理后数据集 |
| `outputs_lie/qwen2.5-1.5b/lie-detector/Questions1000/*.joblib` | 训练好的检测器模型 |
| `outputs_lie/qwen2.5-1.5b/layer_AIE_lie.json` | 因果效应 JSON（if_plot=true 时） |
| `outputs_lie/qwen2.5-1.5b/Questions1000/*.pdf` | 因果效应可视化图（if_plot=true 时） |

---

## 七、运行命令

```powershell
cd d:/CodeProject/LLMScan
$env:HF_HOME = "D:\huggingface_cache\huggingface"
$env:HF_ENDPOINT = "https://hf-mirror.com"
python d:/CodeProject/LLMScan/public_func/causality_analysis.py
```

---

## 八、待办事项

- [ ] 跑完检测器训练和评估，输出 ACC/ROC/F1 指标
- [ ] 验证 `if_plot=True` 时可正常输出因果效应图和 JSON
- [ ] 测试其他任务类型（bias、toxic、jailbreak）
- [ ] 测试其他模型（Llama-3.1、Mistral）需先申请 HuggingFace 授权
- [ ] `causality_analysis_combine.py` 和 `causality_analysis_prompt.py` 中的同类 bug 需要同步修复
