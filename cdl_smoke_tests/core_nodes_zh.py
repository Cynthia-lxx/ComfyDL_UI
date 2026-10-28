"""Chinese display names for the ComfyDL *core* Network & Layers nodes.

The official frontend ships translations for the upstream diffusion nodes
(``model/*``, ``3d``) but knows nothing about the 75 Network & Layers nodes
ComfyDL_UI rehydrated into ``comfy_extras`` — those keep showing English
titles on the canvas.  This table is the Chinese source of truth for them;
``gen_locales.py`` merges it into ``locales/zh/nodeDefs.json`` and
``app/frontend_patch.py`` bakes the merged file into the frontend's zh
``nodeDefs`` chunk (the 0.3.14 frontend never calls the ``/i18n`` endpoint,
so a locale file alone would never be loaded).
"""

CORE_NODE_NAMES_ZH = {
    # --- Activation (14) ---------------------------------------------------
    "ActivationELU": "激活（ELU）",
    "ActivationGELU": "激活（GELU）",
    "ActivationHardSwish": "激活（HardSwish）",
    "ActivationIdentity": "激活（恒等）",
    "ActivationLeakyReLU": "激活（LeakyReLU）",
    "ActivationMish": "激活（Mish）",
    "ActivationReLU": "激活（ReLU）",
    "ActivationReLU6": "激活（ReLU6）",
    "ActivationSELU": "激活（SELU）",
    "ActivationSiLU": "激活（SiLU）",
    "ActivationSigmoid": "激活（Sigmoid）",
    "ActivationSoftmax": "激活（Softmax）",
    "ActivationSoftplus": "激活（Softplus）",
    "ActivationTanh": "激活（Tanh）",
    # --- Basic (8) ----------------------------------------------------------
    "BasicAdd": "基础运算（加）",
    "BasicBroadcast": "基础运算（广播）",
    "BasicConcat": "基础运算（拼接）",
    "BasicEmbedding": "基础运算（嵌入查表）",
    "BasicFlatten": "基础运算（展平）",
    "BasicLinear": "基础运算（线性层）",
    "BasicMultiply": "基础运算（乘）",
    "BasicReshape": "基础运算（重塑）",
    # --- Convolution / upsampling (5) ---------------------------------------
    "ConvolutionConv": "卷积（Conv）",
    "ConvolutionConvTranspose": "卷积（转置卷积）",
    "ConvolutionPixelShuffle": "卷积（像素重排 PixelShuffle）",
    "ConvolutionPixelUnshuffle": "卷积（逆像素重排）",
    "ConvolutionUpsample": "卷积（上采样）",
    # --- Attention (8) -------------------------------------------------------
    "AttentionCausalMask": "注意力（因果掩码）",
    "AttentionCross": "注意力（交叉注意力）",
    "AttentionMultihead": "注意力（多头注意力）",
    "AttentionPaddingMask": "注意力（填充掩码）",
    "AttentionPositionalEncoding": "注意力（位置编码）",
    "AttentionSelf": "注意力（自注意力）",
    "TransformerDecoderBlock": "Transformer 解码器块",
    "TransformerEncoderBlock": "Transformer 编码器块",
    # --- Normalization (7) ---------------------------------------------------
    "NormalizationBatchNorm": "归一化（批归一化）",
    "NormalizationGroupNorm": "归一化（组归一化）",
    "NormalizationInstanceNorm": "归一化（实例归一化）",
    "NormalizationLayerNorm": "归一化（层归一化）",
    "NormalizationRMSNorm": "归一化（RMSNorm）",
    "NormalizationSpectralNorm": "归一化（谱归一化）",
    "NormalizationWeightNorm": "归一化（权重归一化）",
    # --- Pooling (2) ----------------------------------------------------------
    "PoolingAdaptive": "池化（自适应池化）",
    "PoolingSliding": "池化（滑动窗口池化）",
    # --- Recurrent (3) ----------------------------------------------------------
    "RecurrentGRU": "循环网络（GRU）",
    "RecurrentLSTM": "循环网络（LSTM）",
    "RecurrentRNN": "循环网络（RNN）",
    # --- Regularization (1) ------------------------------------------------------
    "RegularizationDropout": "正则化（Dropout）",
    # --- Text (4) ------------------------------------------------------------
    "TextDecode": "文本（解码）",
    "TextEncode": "文本（编码）",
    "TextSlidingWindow": "文本（滑动窗口）",
    "TextVocabBuild": "文本（词表构建）",
    # --- Training family (23) ------------------------------------------------
    "TrainingEvaluate": "训练（评估）",
    "TrainingLRScheduler": "训练（学习率调度器）",
    "TrainingLoadParameters": "训练（加载参数）",
    "TrainingLoop": "训练（训练循环）",
    "TrainingLoss": "训练（损失函数）",
    "TrainingMetrics": "训练（评估指标）",
    "TrainingMode": "训练（训练模式）",
    "TrainingOptimizer": "训练（优化器）",
    "TrainingParameters": "训练（可学习参数）",
    "TrainingParametersExtract": "训练（参数提取）",
    "TrainingParametersMerge": "训练（参数合并）",
    "TrainingParametersToText": "训练（参数转文本）",
    "TrainingRunStats": "训练（运行统计）",
    "TrainingSaveParameters": "训练（保存参数）",
    "TrainingTextToParameters": "训练（文本转参数）",
    # --- Language model pipeline (8) -----------------------------------------
    "LanguageModelBuild": "语言模型（构建）",
    "LanguageModelEmbedding": "语言模型（嵌入）",
    "LanguageModelForward": "语言模型（前向）",
    "LanguageModelGenerate": "语言模型（生成）",
    "LanguageModelLoad": "语言模型（加载）",
    "LanguageModelSave": "语言模型（保存）",
    "LanguageModelTrain": "语言模型（训练）",
    "LanguageModelTransformerBlock": "语言模型（Transformer 块）",
}
