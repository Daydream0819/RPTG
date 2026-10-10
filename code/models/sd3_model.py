import os
import torch
from diffusers import StableDiffusion3Pipeline

# 优化环境：减少不必要的 CPU 线程抢占
os.environ["OMP_NUM_THREADS"] = "1"

class SD3TransformerWrapper:
    def __init__(self, device="cuda", dtype=torch.bfloat16, model_path="/root/autodl-tmp/hf_cache"):
        """
        针对 RTX 5090 优化的 SD3.5 包装器
        - device: 建议 "cuda"
        - dtype: 建议 torch.bfloat16 (Ada架构 5090 原生加速)
        """
        self.device = device
        self.dtype = dtype
        self.model_path = model_path
        
        print(f"📦 正在从 {model_path} 加载 SD3.5 全量模型...")
        
        # 1. 加载 Pipeline
        # 注意：这里先不直接 .to(device)，我们在内部通过动态管理来节省显存
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            local_files_only=True
        )

        # 2. 组件分离与精度管理
        # VAE 必须强制 float32 以避免 SD3.5 常见的“黑图”或“溢出色点”问题
        self.vae = self.pipe.vae.to(device=self.device, dtype=torch.float32)
        
        # Transformer 是 5090 的主战场，直接常驻显存
        self.transformer = self.pipe.transformer.to(device=self.device, dtype=self.dtype)
        
        # 3. 5090 算力全开选项 (可选)
        # 如果你希望生成速度达到极致，取消下面这行注释。首次运行会编译约 2 分钟。
        # self.transformer = torch.compile(self.transformer, mode="reduce-overhead")

        # 4. 显存状态标记
        self.text_encoders_on_gpu = False

    def encode_prompt(self, prompt, negative_prompt=""):
        """
        对齐官方 T5-XXL + CLIP 编码逻辑
        采用了显存“即用即走”策略，防止 32GB 被文本编码器长期挤占
        """
        # 如果编码器在 CPU，则搬运到 GPU
        if not self.text_encoders_on_gpu:
            self.pipe.text_encoder.to(self.device)
            self.pipe.text_encoder_2.to(self.device)
            self.pipe.text_encoder_3.to(self.device)
            self.text_encoders_on_gpu = True

        with torch.no_grad():
            # SD3.5 必须显式设置 max_sequence_length=256，否则提示词理解力大幅下降
            output = self.pipe.encode_prompt(
                prompt=prompt,
                prompt_2=prompt,
                prompt_3=prompt,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt,
                negative_prompt_3=negative_prompt,
                device=self.device,
                do_classifier_free_guidance=True,
                max_sequence_length=256 
            )
        
        # 🚀 显存优化：完成编码后立即将编码器踢回 CPU，为采样循环腾出空间
        self.pipe.text_encoder.to("cpu")
        self.pipe.text_encoder_2.to("cpu")
        self.pipe.text_encoder_3.to("cpu")
        self.text_encoders_on_gpu = False
        torch.cuda.empty_cache() # 强制回收碎片

        # 返回顺序：(prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds)
        return [x.to(self.dtype) for x in output]

    def __call__(self, x, t, context):
        """
        Transformer 前向传播：执行 Batch 并行化 CFG
        x: 输入 Latent [Batch, 16, H/8, W/8]
        t: 时间步标量 (0.0 ~ 1.0)
        context: encode_prompt 返回的特征列表
        """
        # 1. 类型对齐
        x = x.to(device=self.device, dtype=self.dtype)
        
        # 2. 解包文本特征
        p_emb, n_emb, p_pool, n_pool = context

        # 3. 处理时间步 t：SD3 内部使用 0-1000 刻度
        # 我们支持传入标量或 Tensor
        t_val = t.item() if isinstance(t, torch.Tensor) else t
        # 对于并行 CFG，时间步需要复制一份
        t_model = torch.full((x.shape[0] * 2,), t_val * 1000.0, 
                             device=self.device, dtype=self.dtype)

        # 4. 🚀 核心优化：并行计算 (Batch 并行)
        # 拼接顺序：[无条件, 有条件]，这与后续的 chunk(2) 对应
        x_in = torch.cat([x, x], dim=0)
        emb_in = torch.cat([n_emb, p_emb], dim=0)
        pool_in = torch.cat([n_pool, p_pool], dim=0)

        with torch.no_grad():
            # 5090 核心算力点
            v_pred = self.transformer(
                hidden_states=x_in,
                timestep=t_model,
                encoder_hidden_states=emb_in,
                pooled_projections=pool_in,
                return_dict=False
            )[0]

        # 5. 劈开结果：(Unconditional, Conditional)
        v_uncond, v_cond = v_pred.chunk(2, dim=0)
        
        return v_uncond.to(self.dtype), v_cond.to(self.dtype)
