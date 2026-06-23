'''
Hidden-State Detector for Jailbreak/Toxic Detection

核心思路：不做因果干预（AIE），直接用 LLM 正常前向传播的 hidden states 训练检测器。

流程：
  input prompt → tokenizer → LLM 前向传播一次（output_hidden_states=True）
  → 提取最后 N 层 hidden states → last-token pooling → 拼接为固定长度特征向量
  → 训练分类器（LR/MLP）→ 评估（ACC/F1/ROC-AUC）+ PCA/t-SNE 可视化

对比 LLMScan AIE 方法：免去逐层短路干预，每条样本只需 1 次前向传播（vs 27 次）。
'''

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.modelUtils import *
from utils.utils import *
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免 GUI 报错
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import random
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
from joblib import dump

from lllm.classification_utils import Classifier
from lllm.questions_loaders import AutoDAN, GCG, PAP

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


# ============================================================
# 1. 数据集采样
# ============================================================

def sample_balanced_dataset(dataset, max_samples=None, random_state=42):
    """
    从 jailbreak 数据集中平衡采样 adv / non_adv。

    参数:
        dataset: DataFrame，有 'questions' 和 'label' 列
        max_samples: int or None, 总采样数（各取 half）
        random_state: 随机种子

    返回:
        prompts: list of str (已包装好的 prompt)
        labels:  list of int (1=adv, 0=non_adv)
    """
    if max_samples is not None:
        half = max_samples // 2
        adv_idx = dataset[dataset['label'] == 'adv_data'].index[:half]
        non_adv_idx = dataset[dataset['label'] == 'non_adv_data'].index[:half]
        selected = list(adv_idx) + list(non_adv_idx)
        sampled = dataset.loc[selected].sample(frac=1, random_state=random_state).reset_index(drop=True)
        print(f"--> 平衡采样 {len(sampled)} 条 ({half} adv + {len(sampled) - half} non_adv)")
    else:
        sampled = dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)
        print(f"--> 使用全部 {len(sampled)} 条数据")

    prompts = []
    labels = []
    for _, row in sampled.iterrows():
        question = row['questions']
        prompt = prepare_prompt(question)  # utils.utils.prepare_prompt
        prompts.append(prompt)
        labels.append(1 if row['label'] == 'adv_data' else 0)

    print(f"     adv: {sum(labels)} 条, non_adv: {len(labels) - sum(labels)} 条")
    return prompts, labels


# ============================================================
# 2. Hidden State 提取
# ============================================================

def extract_hidden_states(prompts, mt, n_last_layers=5, device='cuda:0'):
    """
    对一批 prompt 做一次前向传播，返回最后 n_last_layers 层的 hidden states。

    参数:
        prompts: list of str
        mt: ModelAndTokenizer 实例
        n_last_layers: 取最后几层
        device: 'cuda:0' or 'cpu'

    返回:
        hidden_states_last_n: list of Tensor, 每个 shape [B, S, hidden_dim]
        attention_mask: Tensor, shape [B, S]
    """
    inputs = make_inputs(mt.tokenizer, prompts, device=device)
    # make_inputs 做左 padding，所以每个序列最后一个位置一定是真实 token

    torch.cuda.empty_cache()
    with torch.no_grad():
        outputs = mt.model(**inputs, output_hidden_states=True)

    # hidden_states: tuple of (embedding + layer_0 + ... + layer_{L-1}) = L+1 个 tensor
    # 每个 tensor shape: [B, S, hidden_dim]
    all_hidden = outputs.hidden_states
    hidden_states_last_n = list(all_hidden[-n_last_layers:])

    # 释放不再需要的
    del outputs, all_hidden, inputs['input_ids']
    torch.cuda.empty_cache()

    attention_mask = inputs['attention_mask']
    return hidden_states_last_n, attention_mask


# ============================================================
# 3. 特征构造：last-token pooling + 拼接
# ============================================================

def construct_features(hidden_states_last_n, attention_mask):
    """
    每层取最后一个非 padding token 的 hidden state，拼接为固定长度向量。

    参数:
        hidden_states_last_n: list of Tensor [B, S, hidden_dim]
        attention_mask: Tensor [B, S]

    返回:
        features: np.ndarray, shape [B, n_layers * hidden_dim]
    """
    B = attention_mask.shape[0]
    # 最后一个非 padding token 的位置 = attention_mask 中 1 的个数 - 1
    last_positions = attention_mask.sum(dim=1) - 1  # shape [B]

    layer_features = []
    for hs in hidden_states_last_n:
        # hs: [B, S, hidden_dim]
        last_token_hs = hs[torch.arange(B, device=hs.device), last_positions, :]  # [B, hidden_dim]
        layer_features.append(last_token_hs.cpu())

    features = torch.cat(layer_features, dim=1)  # [B, n_layers * hidden_dim]
    return features.numpy()


# ============================================================
# 4. 训练与评估
# ============================================================

def train_evaluate(features, labels, test_size=0.3, random_state=42):
    """
    训练 LR + MLP 分类器并评估。

    返回:
        results: dict, 包含各分类器的 {acc, f1, auc, fpr}
    """
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size,
        random_state=random_state, stratify=labels
    )
    print(f"--> 训练集: {X_train.shape[0]} 条, 测试集: {X_test.shape[0]} 条")
    print(f"    特征维度: {X_train.shape[1]}")

    results = {}

    # --- Logistic Regression ---
    print("\n========== Logistic Regression ==========")
    clf_lr = Classifier(X_train, y_train, classifier="logistic", scale=True,
                        max_iter=1000, random_state=random_state)
    acc, auc, conf_matrix, y_pred, y_proba = clf_lr.evaluate(X_test, y_test, return_ys=True)
    f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    print(f"    ACC: {acc:.4f}  F1: {f1:.4f}  ROC-AUC: {auc:.4f}  FPR: {fpr:.4f}")
    results['logistic'] = {'acc': acc, 'f1': f1, 'auc': auc, 'fpr': fpr, 'model': clf_lr}

    # --- MLP ---
    print("\n========== MLP Classifier ==========")
    clf_mlp = Classifier(X_train, y_train, classifier="MLP", scale=True,
                         hidden_layer_sizes=(100,), max_iter=500, random_state=random_state)
    acc2, auc2, cm2, y_pred2, y_proba2 = clf_mlp.evaluate(X_test, y_test, return_ys=True)
    f1_2 = f1_score(y_test, y_pred2)
    tn2, fp2, fn2, tp2 = cm2.ravel()
    fpr2 = fp2 / (fp2 + tn2) if (fp2 + tn2) > 0 else 0.0
    print(f"    ACC: {acc2:.4f}  F1: {f1_2:.4f}  ROC-AUC: {auc2:.4f}  FPR: {fpr2:.4f}")
    results['mlp'] = {'acc': acc2, 'f1': f1_2, 'auc': auc2, 'fpr': fpr2, 'model': clf_mlp}

    return results


# ============================================================
# 5. PCA / t-SNE 可视化
# ============================================================

def visualize_pca_tsne(features, labels, dataset_name, model_name, saving_dir):
    """
    PCA + t-SNE 二维降维可视化，保存为 PDF。
    """
    labels_arr = np.array(labels)
    n_samples = len(labels_arr)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- PCA ---
    pca = PCA(n_components=2, random_state=42)
    features_pca = pca.fit_transform(features)
    for lbl, name, marker in [(0, 'non_adv', 'o'), (1, 'adv', '^')]:
        mask = labels_arr == lbl
        axes[0].scatter(features_pca[mask, 0], features_pca[mask, 1],
                        label=name, marker=marker, alpha=0.6, s=40)
    axes[0].set_title(f'PCA ({dataset_name} | {model_name})\n'
                      f'Var ratio: {pca.explained_variance_ratio_[0]:.3f}, {pca.explained_variance_ratio_[1]:.3f}')
    axes[0].legend()
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')

    # --- t-SNE ---
    perplexity = min(30, n_samples - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    features_tsne = tsne.fit_transform(features)
    for lbl, name, marker in [(0, 'non_adv', 'o'), (1, 'adv', '^')]:
        mask = labels_arr == lbl
        axes[1].scatter(features_tsne[mask, 0], features_tsne[mask, 1],
                        label=name, marker=marker, alpha=0.6, s=40)
    axes[1].set_title(f't-SNE ({dataset_name} | {model_name})\nperplexity={perplexity}')
    axes[1].legend()
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')

    plt.tight_layout()

    # 保存
    fig_dir = os.path.join(saving_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, f"hidden_state_{dataset_name}_{model_name}.pdf")
    plt.savefig(fig_path, bbox_inches="tight")
    print(f"\n--> 可视化已保存: {fig_path}")
    plt.close(fig)


# ============================================================
# 6. 主流程
# ============================================================

def run_hidden_state_detection(dataset, mt, model_name, saving_dir,
                                n_last_layers=5, max_samples=None,
                                test_size=0.3, random_state=42,
                                if_visualize=True):
    """
    完整 pipeline: 采样 → 提取 hidden states → 构造特征 → 训练评估 → 可视化
    """
    dataset_name = dataset.__class__.__name__
    print(f"\n{'='*60}")
    print(f"  Hidden-State Detector: {dataset_name} @ {model_name}")
    print(f"  最后 {n_last_layers} 层 | last-token pooling | max_samples={max_samples}")
    print(f"{'='*60}")

    # Step 1: 采样
    print("\n[1/4] 数据采样...")
    prompts, labels = sample_balanced_dataset(dataset, max_samples=max_samples, random_state=random_state)

    # Step 2: 提取 hidden states
    print(f"\n[2/4] 提取 hidden states（最后 {n_last_layers} 层）...")
    start = time.time()
    hidden_states, attention_mask = extract_hidden_states(prompts, mt, n_last_layers=n_last_layers)
    elapsed = time.time() - start
    print(f"    耗时: {elapsed:.1f}s ({elapsed/len(prompts):.2f}s/sample)")

    # Step 3: 构造特征
    print("\n[3/4] 构造特征...")
    features = construct_features(hidden_states, attention_mask)
    print(f"    特征矩阵: {features.shape}")

    # Step 4: 训练评估
    print("\n[4/4] 训练分类器...")
    results = train_evaluate(features, labels, test_size=test_size, random_state=random_state)

    # 保存模型
    os.makedirs(saving_dir, exist_ok=True)
    for name, r in results.items():
        if r['model'] is not None:
            dump_path = os.path.join(saving_dir, f"{name}_hidden_state_{dataset_name}.joblib")
            dump(r['model'], dump_path)

    # 可视化
    if if_visualize:
        visualize_pca_tsne(features, labels, dataset_name, model_name, saving_dir)

    # 输出汇总
    print(f"\n{'='*60}")
    print(f"  结果汇总")
    print(f"{'='*60}")
    print(f"  {'分类器':<15} {'ACC':>8} {'F1':>8} {'ROC-AUC':>8} {'FPR':>8}")
    print(f"  {'-'*47}")
    for name in ['logistic', 'mlp']:
        r = results[name]
        print(f"  {name:<15} {r['acc']:>8.4f} {r['f1']:>8.4f} {r['auc']:>8.4f} {r['fpr']:>8.4f}")

    return results


# ============================================================
# 7. 命令行入口
# ============================================================

def load_parameters(file_path):
    with open(file_path, 'r') as file:
        parameters = json.load(file)
    return parameters


if __name__ == '__main__':
    import argparse

    current_dir = os.getcwd()
    json_file_path = os.path.join(current_dir, 'public_func', 'parameters.json')
    parameters = load_parameters(json_file_path)

    parser = argparse.ArgumentParser(description='Hidden-State Detector')
    parser.add_argument('--dataset', type=str, help='数据集')
    parser.add_argument('--task', type=str, help='任务类型')
    parser.add_argument('--model_path', type=str, help='模型路径前缀')
    parser.add_argument('--model_name', type=str, help='模型名称')
    parser.add_argument('--saving_dir', type=str, help='输出目录')
    parser.add_argument('--max_samples', type=int, help='最大样本数')
    parser.add_argument('--n_last_layers', type=int, help='最后几层')
    args = parser.parse_args()

    # 命令行覆盖
    if args.model_path:
        parameters['model_path'] = args.model_path
    if args.model_name:
        parameters['model_name'] = args.model_name
    if args.dataset:
        parameters['dataset'] = args.dataset
    if args.saving_dir:
        parameters['saving_dir'] = args.saving_dir
    if args.max_samples:
        parameters['max_samples'] = args.max_samples
    if args.n_last_layers:
        parameters['n_last_layers'] = args.n_last_layers

    print("--> 参数:", parameters)

    model_path = parameters['model_path']
    model_name = parameters['model_name']
    saving_dir = parameters.get('saving_dir', 'outputs_hiddenstate/')
    n_last_layers = parameters.get('n_last_layers', 5)
    max_samples = parameters.get('max_samples', None)
    test_size = parameters.get('test_size', 0.3)
    random_state = parameters.get('random_state', 42)
    if_visualize = parameters.get('if_visualize', True)

    # 加载模型
    print(f"--> 加载模型: {model_path}{model_name}")
    mt = ModelAndTokenizer(
        model_path + model_name,
        low_cpu_mem_usage=True,
        device='cuda:0'
    )
    mt.model
    print("--> 模型加载成功")
    print(f"    层数: {mt.num_layers}, hidden_size: {mt.model.config.hidden_size}")

    # 加载数据集
    dataset = eval(parameters['dataset'])
    print(f"--> 数据集: {parameters['dataset']}, 共 {len(dataset)} 条")

    # 运行
    run_hidden_state_detection(
        dataset=dataset,
        mt=mt,
        model_name=model_name,
        saving_dir=saving_dir,
        n_last_layers=n_last_layers,
        max_samples=max_samples,
        test_size=test_size,
        random_state=random_state,
        if_visualize=if_visualize,
    )
