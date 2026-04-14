# 蚂蚁集团 - 多模态视频理解与交互算法专家 面试准备资料

> 职位编号：GP25032103960714  
> 工作地点：杭州  
> 学历要求：本科  
> 年限要求：2 年以上  
> 生成时间：2026-03-25

---

## 📋 一、岗位深度分析

### 1.1 团队定位

```
蚂蚁集团-CTO-基础智能技术部-多模态认知-多模态交互
│
├── 核心方向：多模态大模型预训练
├── 关键技术：视频问答、音视频交互
├── 目标：构建图像 + 视频 + 语音多模态通用大模型
└── 落地：音视频交互推理加速框架
```

### 1.2 核心能力要求

| 能力维度 | 具体要求 | 面试考察点 |
|----------|----------|------------|
| **基础理论** | 计算机视觉基础理论和方法 | CV 基础知识、深度学习原理 |
| **工程能力** | PyTorch 框架、前沿模型实现 | 代码能力、模型复现 |
| **研究能力** | 学术调研、前沿探索 | 论文阅读、技术敏感度 |
| **项目经验** | 2 年 + 视频算法相关 | 项目深度、落地能力 |
| **加分项** | 顶会论文、大模型经验 | 科研产出、分布式训练 |

### 1.3 技术栈映射

```
┌─────────────────────────────────────────────────────────────┐
│                    岗位技术栈                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  多模态模型                                                  │
│  ├── CLIP/BLIP/LLaVA 等 VLM                                 │
│  ├── Video-LLaVA/Video-ChatGPT 等视频 VLM                   │
│  └── Whisper/AV-HuBERT 等音频模型                           │
│                                                             │
│  视频理解                                                    │
│  ├── 动作识别 (I3D, SlowFast, TimeSformer)                  │
│  ├── 视频问答 (Video-QA)                                    │
│  ├── 时序定位 (Temporal Localization)                       │
│  └── 视频 captioning                                        │
│                                                             │
│  大模型技术                                                  │
│  ├── Transformer 架构                                        │
│  ├── 分布式训练 (DeepSpeed, FSDP)                           │
│  ├── 参数高效微调 (LoRA, QLoRA)                             │
│  └── 推理加速 (TensorRT, vLLM)                              │
│                                                             │
│  工程框架                                                    │
│  ├── PyTorch                                                │
│  ├── MMAction2/MMVid                                        │
│  └── HuggingFace Transformers                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📚 二、核心技术知识点

### 2.1 视频理解基础

#### 关键概念

| 概念 | 说明 | 代表方法 |
|------|------|----------|
| **时空特征** | 同时建模空间（图像）和时间（帧间）信息 | 3D CNN, Transformer |
| **时序建模** | 捕获帧间依赖关系 | RNN, Transformer, TCN |
| **多尺度时序** | 不同时间粒度的特征 | Multi-scale Temporal |
| **长时序建模** | 处理长视频序列 | Sparse Attention, Memory |

#### 经典模型对比

| 模型 | 年份 | 核心思想 | 优点 | 缺点 |
|------|------|----------|------|------|
| **I3D** | CVPR 2017 | 3D CNN，ImageNet 预训练 Kinetics 微调 | 简单有效 | 计算量大 |
| **SlowFast** | ICCV 2019 | 双路径（慢速空间 + 快速时序） | 效率与精度平衡 | 双分支设计复杂 |
| **TimeSformer** | ICML 2021 | 纯 Transformer，时空分离注意力 | 长程依赖好 | 计算复杂度 O(n²) |
| **VideoMAE** | NeurIPS 2022 | 掩码自编码预训练 | 自监督 SOTA | 需要大量预训练 |
| **Uniformer** | CVPR 2022 | CNN+Transformer 混合 | 兼顾局部与全局 | - |

---

### 2.2 多模态视频大模型

#### 主流架构对比

| 模型 | 机构 | 视觉编码器 | 语言模型 | 对齐方式 | 特点 |
|------|------|------------|----------|----------|------|
| **Video-LLaVA** | 北大 | CLIP ViT | LLaMA | MLP Projector | 图像 + 视频统一处理 |
| **Video-ChatGPT** | 达摩院 | CLIP ViT | Vicuna | Q-Former | 指令微调视频对话 |
| **LLaMA-VID** | 商汤 | CLIP ViT | LLaMA | Perceiver | 长视频理解 |
| **Video-LLaMA** | 阿里 | 冻结 ViT | LLaMA | 视频 Q-Former | 音视频双模态 |
| **Chat-UniVi** | 清华 | ViT | Vicuna | 统一视觉 Token | 图像视频统一 |
| **InternVL** | 商汤 | 自研 ViT | InternLM | MLP | 高分辨率支持 |

#### Video-LLaVA 架构详解

```
┌─────────────────────────────────────────────────────────────┐
│                    Video-LLaVA 架构                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   视频输入 → 帧采样 (8-16 帧) → CLIP ViT → 视觉特征          │
│                                              ↓              │
│   图像输入 → CLIP ViT → 视觉特征 ────────────┤              │
│                                              ↓              │
│                                    视觉语言对齐层            │
│                                    (MLP Projector)          │
│                                              ↓              │
│   文本输入 → Token Embedding ───────→ 特征拼接              │
│                                              ↓              │
│                                          LLaMA LLM          │
│                                              ↓              │
│                                          文本输出           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**核心创新点：**
1. **统一视觉编码器**：图像和视频使用相同的 CLIP ViT
2. **高斯掩码策略**：视频帧采样时考虑时间分布
3. **两阶段训练**：先图像指令微调，再视频指令微调

---

### 2.3 音视频交互技术

#### 音频特征提取

| 模型 | 类型 | 特点 | 适用场景 |
|------|------|------|----------|
| **VGGish** | CNN | Google 开源，轻量 | 通用音频分类 |
| **PANNs** | CNN | 预训练模型库 | 音频事件检测 |
| **Whisper** | Transformer | 多语言 ASR | 语音识别 |
| **AV-HuBERT** | Transformer | 唇读 + 语音 | 音视频联合 |

#### 音视频融合方式

```python
# 1. 早期融合（特征级）
def early_fusion(video_feat, audio_feat):
    # 拼接后融合
    combined = torch.cat([video_feat, audio_feat], dim=-1)
    fused = fusion_network(combined)
    return fused

# 2. 晚期融合（决策级）
def late_fusion(video_pred, audio_pred):
    # 加权平均
    fused = α * video_pred + (1-α) * audio_pred
    return fused

# 3. 交叉注意力融合
def cross_modal_attention(video_feat, audio_feat):
    # 视频 Query，音频 Key/Value
    attn_out = CrossAttention(video_feat, audio_feat, audio_feat)
    return attn_out
```

---

### 2.4 分布式训练技术

#### DeepSpeed ZeRO 配置

```json
{
    "train_batch_size": 128,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 2e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8
        }
    },
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "initial_scale_power": 16
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "reduce_scatter": true
    }
}
```

#### 训练效率优化对比

| 优化技术 | 显存节省 | 速度影响 | 适用场景 |
|----------|----------|----------|----------|
| **FP16 混合精度** | 50% | +30% | 通用 |
| **Gradient Checkpointing** | 60% | -20% | 显存受限 |
| **ZeRO-1** | 4-8x | -10% | 大模型 |
| **ZeRO-2** | 8-16x | -15% | 大模型 |
| **ZeRO-3** | 16-64x | -25% | 超大模型 |
| **Offload** | 额外 2-4x | -30% | 单卡训练 |

---

## 🎯 三、面试题库与参考答案

### 3.1 基础知识题

#### Q1: 请解释 3D CNN 与 2D CNN+RNN 在视频理解中的区别

**参考答案：**

| 对比维度 | 3D CNN | 2D CNN+RNN |
|----------|--------|------------|
| **时空建模** | 联合建模（3D 卷积核） | 分离建模（空间 + 时序） |
| **感受野** | 局部时空邻域 | 空间局部 + 时间全局 |
| **计算效率** | 高（并行） | 低（RNN 串行） |
| **长程依赖** | 有限（卷积核大小限制） | 较好（RNN/LSTM） |
| **代表工作** | I3D, C3D, SlowFast | CNN+LSTM, CNN+GRU |

**3D CNN 卷积操作：**
```python
# 3D 卷积：(C_in, T, H, W) → (C_out, T', H', W')
conv3d = nn.Conv3d(in_channels=3, out_channels=64, 
                   kernel_size=(3, 7, 7), stride=(1, 2, 2), 
                   padding=(1, 3, 3))

# 同时捕获空间和时间特征
output = conv3d(input_video)  # input_video: (B, 3, T, H, W)
```

**实际应用建议：**
- 短视频、动作识别 → 3D CNN（I3D, SlowFast）
- 长视频、需要全局时序 → Transformer（TimeSformer）
- 资源受限 → 2D CNN+ 轻量时序模块

---

#### Q2: TimeSformer 的时空分离注意力是如何工作的？

**参考答案：**

**核心思想：** 将 Self-Attention 分解为空间注意力和时间注意力，降低计算复杂度。

**标准 Self-Attention 复杂度：** O((T×N)²) = O(T²×N²)

**时空分离注意力：**
```python
class DividedSpaceTimeAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x, num_frames, num_patches):
        """
        x: (B, T×N, D) T=帧数，N=每帧 patch 数
        """
        B, TN, D = x.shape
        qkv = self.qkv(x).reshape(B, TN, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, TN, D/H)
        
        # 1. 空间注意力（每帧内部）
        x = x.reshape(B * num_frames, num_patches, D)
        q_s = q.reshape(B * num_frames, self.num_heads, num_patches, self.head_dim)
        k_s = k.reshape(B * num_frames, self.num_heads, num_patches, self.head_dim)
        v_s = v.reshape(B * num_frames, self.num_heads, num_patches, self.head_dim)
        
        attn_s = (q_s @ k_s.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_s = attn_s.softmax(dim=-1)
        x_s = (attn_s @ v_s).transpose(1, 2).reshape(B * num_frames, num_patches, D)
        
        # 2. 时间注意力（同位置 patch 跨帧）
        x_s = x_s.reshape(B, num_frames, num_patches, D).transpose(1, 2)  # (B, N, T, D)
        x_s = x_s.reshape(B * num_patches, num_frames, D)
        
        q_t = q.permute(0, 2, 1, 3).reshape(B * num_patches, self.num_heads, num_frames, self.head_dim)
        k_t = k.permute(0, 2, 1, 3).reshape(B * num_patches, self.num_heads, num_frames, self.head_dim)
        v_t = v.permute(0, 2, 1, 3).reshape(B * num_patches, self.num_heads, num_frames, self.head_dim)
        
        attn_t = (q_t @ k_t.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_t = attn_t.softmax(dim=-1)
        x_t = (attn_t @ v_t).transpose(1, 2).reshape(B * num_patches, num_frames, D)
        
        # 恢复形状
        x_t = x_t.reshape(B, num_patches, num_frames, D).transpose(1, 2)  # (B, T, N, D)
        x_t = x_t.reshape(B, TN, D)
        
        return self.proj(x_t)
```

**复杂度对比：**
- 标准注意力：O(T²×N²)
- 时空分离：O(T×N² + N×T²) = O(T×N² + N×T²)

当 T=16, N=196 时：
- 标准：O(16²×196²) ≈ 9.8M
- 分离：O(16×196² + 196×16²) ≈ 0.66M

**计算量减少约 15 倍！**

---

#### Q3: VideoMAE 的掩码策略与图像 MAE 有什么区别？

**参考答案：**

**图像 MAE 掩码策略：**
- 随机掩码 75% 的 patch
- 每帧独立掩码
- 空间掩码

**VideoMAE 掩码策略：**
- 随机掩码 90% 的 patch（更高掩码率）
- **管状掩码 (Tube Masking)**：时空连续掩码
- 同时掩码同一位置的多帧

```python
def video_masking(video_patches, mask_ratio=0.9, tube_size=4):
    """
    VideoMAE 管状掩码
    
    Args:
        video_patches: (B, T, N, D) T=帧数，N=patch 数
        mask_ratio: 掩码比例
        tube_size: 管状掩码的时间长度
    """
    B, T, N, D = video_patches.shape
    
    # 1. 生成管状掩码
    # 每帧相同位置连续掩码 tube_size 帧
    num_mask_tubes = int(N * mask_ratio / tube_size * T)
    
    # 随机选择掩码的时空位置
    mask_indices = torch.randperm(N)[:num_mask_tubes]
    mask_frames = torch.randint(0, T - tube_size + 1, (num_mask_tubes,))
    
    # 2. 生成掩码矩阵
    mask = torch.ones(B, T, N, dtype=bool)
    for i in range(num_mask_tubes):
        patch_idx = mask_indices[i]
        frame_start = mask_frames[i]
        mask[:, frame_start:frame_start+tube_size, patch_idx] = False
    
    return mask

# 管状掩码的优势：
# 1. 强制模型学习时序冗余
# 2. 更难的预训练任务
# 3. 更好的时序特征表示
```

**掩码率对比：**

| 任务 | 掩码率 | 理由 |
|------|--------|------|
| 图像 MAE | 75% | 平衡难度与信息量 |
| VideoMAE | 90% | 视频有时序冗余，可更高掩码率 |

---

### 3.2 项目经验题

#### Q4: 请介绍一个你做过的视频理解项目

**回答框架 (STAR 法则)：**

```
【Situation 情境】
- 项目背景和目标
- 业务场景和挑战

【Task 任务】
- 你的角色和职责
- 需要解决的核心问题

【Action 行动】
- 技术方案设计
- 模型选型和改进
- 关键实现细节

【Result 结果】
- 量化指标提升
- 业务价值
- 个人贡献
```

**示例回答：**

> **项目背景：** 智能视频监控异常行为检测系统
> 
> **挑战：**
> - 需要实时检测打架、跌倒、聚集等异常行为
> - 视频流 24 小时不间断，计算资源有限
> - 长尾场景多（罕见异常行为样本少）
> 
> **我的角色：** 算法负责人，负责模型设计、训练和部署
> 
> **技术方案：**
> 1. **模型选型**：SlowFast 作为 backbone，平衡精度和速度
> 2. **时序建模**：添加 Transformer 编码器捕获长时序依赖
> 3. **数据增强**：针对长尾场景设计专用增强（遮挡、光照变化）
> 4. **部署优化**：TensorRT 量化，延迟从 500ms 降到 100ms
> 
> **结果：**
> - 异常检测 mAP 从 65% 提升到 82%
> - 推理速度提升 5 倍，支持 16 路视频流实时分析
> - 误报率降低 60%，减少人工审核成本

---

#### Q5: 如何处理视频理解中的长尾场景问题？

**参考答案：**

**长尾场景类型：**
| 类型 | 示例 | 占比 |
|------|------|------|
| 罕见动作 | 跌倒、打架 | <1% |
| 特殊场景 | 夜间、极端天气 | <5% |
| 罕见物体 | 特殊车辆、动物 | <2% |
| 组合场景 | 雨天 + 夜间 + 拥堵 | <0.5% |

**解决方案：**

**1. 数据层面**
```python
# 过采样稀有场景
def oversample_rare_classes(dataset, min_samples=1000):
    class_counts = count_samples_per_class(dataset)
    
    oversampled_data = []
    for class_id, count in class_counts.items():
        class_samples = get_samples_by_class(dataset, class_id)
        
        if count < min_samples:
            # 过采样
            repeat_ratio = min_samples // count
            augmented = [augment(sample) for sample in class_samples * repeat_ratio]
            oversampled_data.extend(augmented)
        else:
            oversampled_data.extend(class_samples)
    
    return oversampled_data
```

**2. 损失函数**
```python
# 类别平衡损失
class BalancedLoss(nn.Module):
    def __init__(self, class_counts):
        super().__init__()
        # 计算类别权重（逆频率）
        total = sum(class_counts)
        self.weights = [total / (len(class_counts) * count) 
                       for count in class_counts]
    
    def forward(self, predictions, labels):
        loss = F.cross_entropy(predictions, labels, reduction='none')
        # 按类别加权
        weighted_loss = loss * torch.tensor(self.weights)[labels]
        return weighted_loss.mean()
```

**3. 迁移学习**
```python
# 从相关任务迁移
# 1. 在大规模动作识别数据集（Kinetics）上预训练
# 2. 在目标域数据上微调
# 3. 使用领域自适应减少分布差异
```

**4. 数据合成**
```python
# 使用仿真/生成模型合成稀有场景
# - 使用 Blender/Unreal 生成罕见动作
# - 使用 Diffusion 模型生成特殊场景
# - 使用 Cut-Paste 合成罕见物体组合
```

---

### 3.3 技术设计题

#### Q6: 如果要设计一个视频问答系统，你会如何设计架构？

**参考答案：**

**系统架构：**
```
┌─────────────────────────────────────────────────────────────┐
│                    视频问答系统架构                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  输入层                                                      │
│  ├── 视频输入 → 帧采样 (8-16 帧)                            │
│  └── 问题输入 → Token 化                                     │
│                                                             │
│  编码层                                                      │
│  ├── 视频编码器 (CLIP ViT / SlowFast)                       │
│  ├── 文本编码器 (BERT / LLM Embedding)                      │
│  └── 音频编码器 (Whisper / VGGish) [可选]                   │
│                                                             │
│  融合层                                                      │
│  ├── Cross-Attention (视频 Query，文本 Key/Value)           │
│  ├── 多模态 Transformer                                     │
│  └── 时序对齐模块                                           │
│                                                             │
│  解码层                                                      │
│  ├── 答案生成 (LLM Decoder)                                 │
│  └── 答案检索 (RAG) [可选]                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**关键模块实现：**

```python
class VideoQAModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # 视频编码器
        self.video_encoder = load_clip_vit()
        
        # 文本编码器
        self.text_encoder = load_bert()
        
        # 多模态融合
        self.fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=8),
            num_layers=6
        )
        
        # 答案生成头
        self.answer_head = nn.Linear(768, config.vocab_size)
    
    def forward(self, video_frames, question_tokens):
        # 1. 视频编码
        video_features = self.video_encoder(video_frames)
        # (B, T×N, D)
        
        # 2. 文本编码
        text_features = self.text_encoder(question_tokens)
        # (B, L, D)
        
        # 3. 多模态融合
        combined = torch.cat([video_features, text_features], dim=1)
        fused = self.fusion(combined)
        
        # 4. 答案生成
        answer_logits = self.answer_head(fused[:, 0])  # 使用 CLS token
        
        return answer_logits
```

**训练策略：**
1. **预训练**：在 How2QA/MSR-VTT 等大规模视频问答数据集上预训练
2. **微调**：在目标领域数据上微调
3. **数据增强**：问题改写、视频裁剪、时序扰动

**评估指标：**
- 准确率 (Accuracy)
- BLEU/ROUGE (生成质量)
- 人工评估 (相关性、准确性)

---

#### Q7: 如何优化长视频的推理效率？

**参考答案：**

**长视频挑战：**
- 1 小时视频 @30fps = 108,000 帧
- 无法全部输入模型
- 需要高效采样和推理策略

**优化方案：**

**1. 关键帧采样**
```python
def keyframe_sampling(video, num_frames=32):
    """基于内容变化的关键帧采样"""
    
    # 计算帧间差异
    frame_diffs = []
    for i in range(1, len(video)):
        diff = compute_similarity(video[i-1], video[i])
        frame_diffs.append(diff)
    
    # 选择变化最大的帧
    peak_indices = find_peaks(frame_diffs, num_peaks=num_frames-1)
    
    # 添加首尾帧
    selected_frames = [0] + list(peak_indices) + [len(video)-1]
    
    return [video[i] for i in selected_frames]
```

**2. 层次化推理**
```python
def hierarchical_video_inference(video, chunk_size=32):
    """层次化视频理解"""
    
    # 1. 分块处理
    chunks = [video[i:i+chunk_size] for i in range(0, len(video), chunk_size)]
    
    # 2. 块级编码
    chunk_features = []
    for chunk in chunks:
        feat = model.encode(chunk)
        chunk_features.append(feat)
    
    # 3. 全局聚合
    global_feature = aggregate(chunk_features)  # Mean/Attention/Transformer
    
    # 4. 基于全局表示推理
    output = model.reason(global_feature)
    
    return output
```

**3. 缓存优化**
```python
class VideoInferenceCache:
    def __init__(self):
        self.frame_cache = {}
        self.feature_cache = {}
    
    def get_or_compute(self, frame_id, model):
        """缓存已计算的特征"""
        if frame_id not in self.feature_cache:
            frame = load_frame(frame_id)
            self.feature_cache[frame_id] = model.encode(frame)
        return self.feature_cache[frame_id]
```

**4. 稀疏注意力**
```python
# 只关注关键区域和关键帧
def sparse_video_attention(features, attention_mask):
    """稀疏时空注意力"""
    
    # 空间稀疏：只关注前景区域
    spatial_mask = detect_foreground(features)
    
    # 时间稀疏：只关注关键帧
    temporal_mask = detect_keyframes(features)
    
    # 联合掩码
    combined_mask = spatial_mask & temporal_mask
    
    # 稀疏注意力计算
    output = sparse_attention(features, mask=combined_mask)
    
    return output
```

**效果对比：**

| 方案 | 帧数 | 延迟 | 精度损失 |
|------|------|------|----------|
| 全量推理 | 108,000 | 3600s | 0% |
| 均匀采样 | 32 | 1s | 15% |
| 关键帧采样 | 32 | 1s | 8% |
| 层次化 + 缓存 | 32+ | 0.5s | 5% |

---

### 3.4 前沿技术题

#### Q8: 你如何看待多模态大模型在视频理解领域的发展趋势？

**参考答案：**

**当前趋势：**

| 趋势 | 说明 | 代表工作 |
|------|------|----------|
| **统一架构** | 图像/视频/音频统一处理 | Video-LLaVA, UniPerceiver |
| **更长上下文** | 支持小时级视频理解 | LLaMA-VID, LongVLM |
| **多模态交互** | 音视频 + 文本联合推理 | Video-LLaMA, AV-LLM |
| **高效推理** | 实时视频流处理 | StreamingLLM, VideoStreaming |
| **具身智能** | 视频理解 + 动作执行 | RT-2, PaLM-E |

**技术挑战：**

1. **计算效率**
   - 视频 token 数量巨大（16 帧×196patch=3136 tokens）
   - 需要更高效的注意力机制

2. **长时序建模**
   - 当前模型只能处理秒级视频
   - 需要分钟/小时级理解能力

3. **细粒度对齐**
   - 视频 - 文本时序对齐困难
   - 需要更精确的时刻定位

4. **评估体系**
   - 缺乏统一的视频 VLM 评测基准
   - 需要更全面的评估指标

**未来方向：**

```
短期 (1-2 年):
├── 更高效的视频编码器
├── 更长上下文支持
└── 更好的时序对齐

中期 (3-5 年):
├── 实时视频流理解
├── 多模态交互对话
└── 具身视频理解

长期 (5+ 年):
├── 通用视频智能
├── 视频世界模型
└── 视频生成 + 理解统一
```

---

#### Q9: 请介绍最近关注的多模态视频理解前沿论文

**参考答案：**

**2024-2025 年重要论文：**

| 论文 | 会议 | 核心贡献 | 启发 |
|------|------|----------|------|
| **Video-LLaVA** | CVPR 2024 | 图像视频统一指令微调 | 统一视觉表示 |
| **LLaMA-VID** | CVPR 2024 | 长视频理解，Perceiver 压缩 | 长序列处理 |
| **Video-ChatGPT** | ACL 2024 | 视频对话指令微调 | 交互式理解 |
| **TimeChat** | CVPR 2024 | 时序定位增强 | 细粒度对齐 |
| **VideoLLaMO** | arXiv 2024 | 多语言视频理解 | 国际化支持 |

**Video-LLaVA 详细解读：**

**核心创新：**
1. **统一视觉编码器**：CLIP ViT 同时处理图像和视频帧
2. **高斯时间掩码**：帧采样时考虑时间分布
3. **两阶段训练**：
   - 阶段 1：图像指令微调（冻结 ViT，训练 Projector+LLM）
   - 阶段 2：视频指令微调（解冻部分 ViT 层）

**实验结果：**
- MSVD-QA: 67.2% → 71.5%
- MSRVTT-QA: 45.8% → 52.3%
- ActivityNet-QA: 32.1% → 38.7%

**可借鉴点：**
- 统一架构简化了系统设计
- 两阶段训练策略可迁移到其他任务
- 高斯掩码可应用于其他时序采样场景

---

### 3.5 编程题

#### Q10: 实现视频帧采样函数

**题目：** 实现一个视频帧采样函数，支持均匀采样、关键帧采样、随机采样三种模式。

```python
import numpy as np
from typing import List, Literal

def sample_video_frames(
    video_length: int,
    num_frames: int,
    mode: Literal['uniform', 'keyframe', 'random'] = 'uniform',
    frame_scores: List[float] = None
) -> List[int]:
    """
    视频帧采样函数
    
    Args:
        video_length: 视频总帧数
        num_frames: 需要采样的帧数
        mode: 采样模式 ('uniform', 'keyframe', 'random')
        frame_scores: 每帧的重要性分数（keyframe 模式需要）
    
    Returns:
        采样帧的索引列表
    
    Examples:
        >>> sample_video_frames(100, 10, 'uniform')
        [0, 11, 22, 33, 44, 55, 66, 77, 88, 99]
        
        >>> sample_video_frames(100, 10, 'random')
        [3, 15, 27, 39, 51, 63, 75, 87, 92, 98]
    """
    if num_frames >= video_length:
        return list(range(video_length))
    
    if mode == 'uniform':
        # 均匀采样
        indices = np.linspace(0, video_length - 1, num_frames, dtype=int)
        return indices.tolist()
    
    elif mode == 'random':
        # 随机采样（不重复）
        indices = np.random.choice(video_length, num_frames, replace=False)
        return sorted(indices.tolist())
    
    elif mode == 'keyframe':
        if frame_scores is None:
            raise ValueError("keyframe 模式需要提供 frame_scores")
        
        if len(frame_scores) != video_length:
            raise ValueError("frame_scores 长度必须等于 video_length")
        
        # 选择分数最高的帧
        indices = np.argsort(frame_scores)[-num_frames:]
        return sorted(indices.tolist())
    
    else:
        raise ValueError(f"不支持的采样模式：{mode}")


# 测试
if __name__ == '__main__':
    # 均匀采样测试
    uniform_result = sample_video_frames(100, 10, 'uniform')
    print(f"均匀采样：{uniform_result}")
    
    # 随机采样测试
    random_result = sample_video_frames(100, 10, 'random')
    print(f"随机采样：{random_result}")
    
    # 关键帧采样测试
    scores = np.random.rand(100)
    keyframe_result = sample_video_frames(100, 10, 'keyframe', scores.tolist())
    print(f"关键帧采样：{keyframe_result}")
```

**复杂度分析：**
- 时间复杂度：O(N log N)（关键帧采样需要排序）
- 空间复杂度：O(N)

---

#### Q11: 实现一个简单的视频分类模型

```python
import torch
import torch.nn as nn
import torchvision.models as models

class SimpleVideoClassifier(nn.Module):
    """
    简单视频分类模型
    2D CNN + 时序池化
    """
    def __init__(self, num_classes=400, backbone='resnet50'):
        super().__init__()
        
        # 2D CNN backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            self.backbone.fc = nn.Identity()
            feature_dim = 2048
        elif backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
            self.backbone.fc = nn.Identity()
            feature_dim = 512
        
        # 时序建模
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        # 分类头
        self.classifier = nn.Linear(feature_dim, num_classes)
    
    def forward(self, video_frames):
        """
        Args:
            video_frames: (B, T, 3, H, W) T=帧数
        Returns:
            logits: (B, num_classes)
        """
        B, T, C, H, W = video_frames.shape
        
        # 1. 逐帧特征提取
        video_frames = video_frames.reshape(B * T, C, H, W)
        frame_features = self.backbone(video_frames)
        # (B*T, feature_dim)
        
        # 2. 恢复时序维度
        frame_features = frame_features.reshape(B, T, -1)
        # (B, T, feature_dim)
        
        # 3. 时序池化
        frame_features = frame_features.transpose(1, 2)
        # (B, feature_dim, T)
        pooled = self.temporal_pool(frame_features).squeeze(-1)
        # (B, feature_dim)
        
        # 4. 分类
        logits = self.classifier(pooled)
        
        return logits


# 测试
if __name__ == '__main__':
    model = SimpleVideoClassifier(num_classes=400)
    
    # 模拟输入：batch=2, 16 帧，224x224
    dummy_input = torch.randn(2, 16, 3, 224, 224)
    
    output = model(dummy_input)
    print(f"输出形状：{output.shape}")  # 应该为 (2, 400)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量：{total_params / 1e6:.2f}M")
```

---

## 💼 四、项目准备建议

### 4.1 项目梳理清单

```
【项目 1：视频理解项目】
□ 项目背景和目标
□ 你的角色和贡献
□ 技术方案设计
□ 核心难点和解决
□ 量化成果（指标提升）
□ 代码/论文/专利产出

【项目 2：多模态项目】
□ 模态类型（图像/视频/文本/音频）
□ 融合方式（早期/晚期/混合）
□ 对齐方法
□ 训练策略
□ 落地效果

【项目 3：大模型项目】（如有）
□ 模型规模
□ 训练框架
□ 分布式配置
□ 优化技术
□ 推理部署
```

### 4.2 技术亮点准备

**建议准备 3-5 个技术亮点：**

| 亮点类型 | 示例 | 准备要点 |
|----------|------|----------|
| **模型改进** | 提出新的注意力机制 | 动机、设计、效果对比 |
| **训练优化** | 分布式训练加速 3 倍 | 技术方案、量化收益 |
| **部署优化** | 模型量化延迟降低 70% | 优化方法、精度影响 |
| **数据创新** | 构建 10 万 + 视频数据集 | 数据规模、标注质量 |
| **开源贡献** | MMAction2 核心贡献者 | 贡献内容、社区影响 |

---

## 🎯 五、面试流程预测

### 5.1 典型面试流程

```
┌─────────────────────────────────────────────────────────────┐
│                    蚂蚁面试流程                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  轮次 1: 技术一面 (60 分钟)                                  │
│  ├── 自我介绍 (5 分钟)                                       │
│  ├── 项目深挖 (20 分钟)                                      │
│  ├── 技术基础 (20 分钟)                                      │
│  └── 编程题 (15 分钟)                                        │
│                                                             │
│  轮次 2: 技术二面 (60 分钟)                                  │
│  ├── 项目技术细节 (20 分钟)                                  │
│  ├── 技术设计题 (25 分钟)                                    │
│  ├── 前沿技术 (10 分钟)                                      │
│  └── 编程题 (15 分钟)                                        │
│                                                             │
│  轮次 3: 主管面 (45 分钟)                                    │
│  ├── 职业规划 (10 分钟)                                      │
│  ├── 团队协作 (10 分钟)                                      │
│  ├── 技术视野 (15 分钟)                                      │
│  └── 反向提问 (10 分钟)                                      │
│                                                             │
│  轮次 4: HR 面 (30 分钟)                                     │
│  ├── 离职原因                                                │
│  ├── 期望薪资                                                │
│  ├── 文化匹配                                                │
│  └── 入职时间                                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 各环节考察重点

| 轮次 | 考察重点 | 准备建议 |
|------|----------|----------|
| **技术一面** | 基础扎实度、代码能力 | 复习 CV 基础、刷 LeetCode |
| **技术二面** | 技术深度、设计能力 | 准备项目细节、系统设计 |
| **主管面** | 技术视野、潜力 | 了解前沿、思考行业 |
| **HR 面** | 稳定性、文化匹配 | 真诚沟通、了解团队 |

---

## 📖 六、推荐学习资源

### 6.1 必读论文

**视频理解基础：**
1. I3D: "Quo Vadis, Action Recognition?" (CVPR 2017)
2. SlowFast: "SlowFast Networks for Video Recognition" (ICCV 2019)
3. TimeSformer: "Is Space-Time Attention All You Need?" (ICML 2021)

**视频大模型：**
1. Video-LLaVA: "Video-LLaVA: Learning United Visual Representation" (CVPR 2024)
2. Video-ChatGPT: "Video-ChatGPT: Towards Detailed Video Understanding" (ACL 2024)
3. LLaMA-VID: "LLaMA-VID: An Image is Worth 2 Tokens in Large Language Models" (CVPR 2024)

**多模态学习：**
1. CLIP: "Learning Transferable Visual Models From Natural Language Supervision" (ICML 2021)
2. BLIP-2: "Bootstrapping Language-Image Pre-training" (ICML 2023)
3. LLaVA: "Visual Instruction Tuning" (NeurIPS 2023)

### 6.2 开源项目

| 项目 | 链接 | 用途 |
|------|------|------|
| MMAction2 | https://github.com/open-mmlab/mmaction2 | 视频理解工具箱 |
| Video-LLaVA | https://github.com/PKU-YuanGroup/Video-LLaVA | 视频大模型 |
| HuggingFace Transformers | https://github.com/huggingface/transformers | 大模型库 |
| DeepSpeed | https://github.com/microsoft/DeepSpeed | 分布式训练 |

### 6.3 数据集

| 数据集 | 规模 | 用途 |
|--------|------|------|
| Kinetics-400/600/700 | 400/600/700 类 | 动作识别 |
| Something-Something V2 | 174 类 | 时序动作 |
| MSR-VTT | 10K 视频 | 视频 caption |
| MSVD-QA | 10K QA 对 | 视频问答 |
| ActivityNet | 20K 视频 | 长视频理解 |
| How2QA | 7K 视频 |  instructional 视频 QA |

---

## 💡 七、面试技巧

### 7.1 回答技巧

**STAR 法则：**
- **S**ituation: 情境背景
- **T**ask: 任务目标
- **A**ction: 行动措施
- **R**esult: 结果成果

**技术回答结构：**
```
1. 直接回答问题核心
2. 展开详细说明
3. 举例/代码佐证
4. 总结关键要点
```

### 7.2 反向提问建议

**可以问的问题：**
- 团队目前的技术重点和挑战？
- 岗位的具体工作内容和期望？
- 团队的技术栈和开发流程？
- 个人成长和发展机会？

**避免问的问题：**
- 薪资福利（HR 面再问）
- 加班情况（可能显得不积极）
- 过于基础的问题（显得准备不足）

---

## 📝 八、面试前检查清单

### 技术准备
- [ ] 复习视频理解基础（3D CNN、Transformer）
- [ ] 熟悉主流视频大模型架构
- [ ] 准备 2-3 个项目的详细说明
- [ ] 刷 LeetCode 中等难度题目 10-20 道
- [ ] 复习 PyTorch 常用操作

### 材料准备
- [ ] 更新简历（突出视频/多模态相关经验）
- [ ] 准备项目代码/论文/专利列表
- [ ] 准备 GitHub/技术博客链接（如有）

### 其他准备
- [ ] 测试面试设备（摄像头、麦克风）
- [ ] 准备安静的面试环境
- [ ] 提前 10 分钟进入面试房间

---

*文档生成时间：2026-03-25*  
*祝面试顺利！🎉*
