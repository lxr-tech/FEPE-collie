# copy from https://github.com/OpenLMLab/collie/blob/dev/collie/models/llama/model.py

import os
import gc
import json

import numpy as np

import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core import tensor_parallel

from deepspeed.pipe import LayerSpec, TiedLayerSpec
from deepspeed.accelerator import get_accelerator

import math
from einops import rearrange

try:
    from flash_attn.modules.mha import FlashSelfAttention as FlashAttention
    from flash_attn.bert_padding import index_first_axis, unpad_input, pad_input
except ModuleNotFoundError:
    FlashAttention = None

from collie.log.logger import logger
from collie.config import load_config
from collie.config import CollieConfig
from collie.utils import progress, env, dict_as_params, concat_tensor
from collie.driver.io import IODriver
from collie.models.base import CollieModelForCausalLM
from collie.module import ColumnParallelLinearWithoutBias, RowParallelLinearWithoutBias, ColumnParallelLMHead

from typing import Union, Optional, Tuple
from collections import OrderedDict
from transformers.modeling_utils import dtype_byte_size
from transformers.modeling_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

# {'exp': True if args.exp == 'xpos' else False, '1d': True if args.dim == '1d' else False, 
#  'imp': True if args.imp == 'imp' else False, 'log': True if args.log == 'log' else False, }


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)
    # return torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x)


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, head_dim: int, config) -> None:
        super().__init__()
        self.pe_config = config.pe_config
        
        # batch_size, seq_len, n_head (fixed for tp), head_dim 
        
        if self.pe_config['imp']:
            order, beta = 3,  1 / math.log(10000.0)  # 0.10856
            start, end = math.pow(0.0001, 1 / order), math.pow(0.9999, 1 / order)  # 
            if self.pe_config['1d']:
                omega = np.power(np.linspace(start, end, head_dim), order)  # * beta * 2
            else:
                omega = np.power(np.linspace(start, end, head_dim // 2), order)   # * beta * 2
                omega = np.concatenate([omega, omega], axis=-1)
            omega = omega[None, None, :]
            expos = (beta / omega) / (1/order * np.power(omega, 1/order - 1))  # * np.power(2 * beta, -1/order))
        else:
            if self.pe_config['1d']:
                omega = 1.0 / (10000.0 ** (np.arange(0, head_dim, 1) / head_dim))
            else:
                omega = 1.0 / (10000.0 ** (np.arange(0, head_dim, 2) / head_dim))
                omega = np.concatenate([omega, omega], axis=-1)
            omega = omega[None, None, :]
            expos = np.ones_like(omega)
        self.register_buffer("omega", torch.tensor(omega), persistent=False)
        self.register_buffer("expos", torch.tensor(np.sqrt(expos)), persistent=False)

        if self.pe_config['exp']:
            if self.pe_config['imp']:
                scale = - np.log(omega) / np.log(10000.0) * head_dim
            else:
                scale = np.arange(0, head_dim, 1 if self.pe_config['1d'] else 2)
                scale = scale if self.pe_config['1d'] else np.concatenate([scale, scale], axis=-1)
                scale = scale[None, None, :]
            scale = (scale + 0.4 * head_dim) / (1.4 * head_dim)
            self.register_buffer("scale", torch.tensor(scale), persistent=False)

        # inv_freq = 1.0 / (10000.0 ** (
        #     torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))
        # self.register_buffer('inv_freq', inv_freq)  # lxr：直接换掉旋转角

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                index_ids):
        dtype = query.dtype
        
        # batch_size, seq_len, n_head (fixed for tp), head_dim 
        
        t = index_ids.reshape(-1, 1, 1).float()
        seq_len = index_ids.max()
        
        theta = self.omega.float() * t
        sin, cos, expos = torch.sin(theta), torch.cos(theta), self.expos.float()

        q, k = query.float() * expos, key.float() * expos

        if self.pe_config['1d']:
            q_real = torch.cat([q * cos, q * sin], dim=-1)
            k_real = torch.cat([k * cos, k * sin], dim=-1)
            # q_imag = torch.cat([q * sin, -q * cos], dim=-1)
        else:
            q_real = q * cos + rotate_half(q) * sin
            k_real = k * cos + rotate_half(k) * sin
            # q_imag = q * sin - rotate_half(q) * cos
            
        if self.pe_config['exp']:
            scale = self.scale ** ((t - seq_len // 2) / self.pe_config['exp_base'])
            scale = scale if not self.pe_config['1d'] else torch.cat([scale, scale], dim=-1)
            q_real = q_real * scale
            k_real = k_real / scale

        if self.pe_config['log']:
            q_real = q_real * torch.log(t+1) / math.log(self.pe_config['log_base'])

        # query = torch.view_as_complex(
        #     query.float().reshape(*query.shape[:-1], -1, 2))
        # key = torch.view_as_complex(
        #     key.float().reshape(*key.shape[:-1], -1, 2))
        # freqs = torch.outer(torch.arange(
        #     (2 ** 16) * 2, device=self.inv_freq.device), self.inv_freq).float()
        # freqs_cis = torch.polar(torch.ones_like(freqs), freqs)[
        #     start_pos: start_pos + seq_len]
        # shape = [d if i == 1 or i == query.ndim -
        #          1 else 1 for i, d in enumerate(query.shape)]
        # freqs_cis = freqs_cis.view(*shape)
        # query = torch.view_as_real(query * freqs_cis).flatten(3)
        # key = torch.view_as_real(key * freqs_cis).flatten(3)  # lxr：1d：可以在这个地方把q k维度扩张1倍；imp：head_dim是并行无关的；log：seq_len+start_pos是真正的t的下标；exp：s下标是从start_pos到start_pos+seq_len
        
        
        return q_real.type(dtype), k_real.type(dtype)

class RMSNormalize(nn.Module):
    def __init__(self, dim=None, dtype=torch.float, eps=1e-5, weight=None):
        super(RMSNormalize, self).__init__()
        if weight is not None:
            self.weight = weight
        else:
            self.weight = nn.Parameter(torch.ones(dim, dtype=dtype, device=get_accelerator().current_device_name()))
        self.eps = eps
    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)
        return hidden_states * self.weight

class LlamaLayer(nn.Module):
    def __init__(self, config: CollieConfig) -> None:
        super().__init__()
        self.config = config
        if hasattr(config, "num_key_value_heads"):
            # llama2 (transformers >= 4.31.0)
            self.num_key_value_heads = config.num_key_value_heads
        else:
            self.num_key_value_heads = config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // self.num_key_value_heads
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = nn.ModuleDict(
            {
                "q_proj": ColumnParallelLinearWithoutBias(
                    config.hidden_size,
                    self.num_heads * self.head_dim,
                    bias=False,
                    gather_output=False,
                    init_method=lambda x: x
                ),
                "k_proj": ColumnParallelLinearWithoutBias(
                    config.hidden_size,
                    self.num_key_value_heads * self.head_dim,
                    bias=False,
                    gather_output=False,
                    init_method=lambda x: x
                ),
                "v_proj": ColumnParallelLinearWithoutBias(
                    config.hidden_size,
                    self.num_key_value_heads * self.head_dim,
                    bias=False,
                    gather_output=False,
                    init_method=lambda x: x
                ),
                "o_proj": RowParallelLinearWithoutBias(
                    self.num_heads * self.head_dim,
                    config.hidden_size,
                    bias=False,
                    input_is_parallel=True,
                    init_method=lambda x: x
                ),
                "rotary_emb": RotaryPositionEmbedding(
                    self.config.hidden_size // self.config.num_attention_heads,
                    config=config
                ),
            }
        )
        self.input_layernorm = RMSNormalize(
            dim=config.hidden_size,
            eps=config.rms_norm_eps
        )
        self.mlp = nn.ModuleDict({
            "gate_proj": ColumnParallelLinearWithoutBias(
                config.hidden_size,
                config.intermediate_size,
                bias=False,
                gather_output=False,
                init_method=lambda x: x
            ),
            "up_proj": ColumnParallelLinearWithoutBias(
                config.hidden_size,
                config.intermediate_size,
                bias=False,
                gather_output=False,
                init_method=lambda x: x
            ),
            "down_proj": RowParallelLinearWithoutBias(
                config.intermediate_size,
                config.hidden_size,
                bias=False,
                input_is_parallel=True,
                init_method=lambda x: x
            )
        })
        self.post_attention_layernorm = RMSNormalize(
            dim=config.hidden_size,
            eps=config.rms_norm_eps
        )
        # 务必保持变量名一致
        self.use_cache = self.config.model_config.use_cache
        self.past_key_values = None
        self.hidden_states = None
        
        self.flash_attn = FlashAttention(softmax_scale=1 / math.sqrt(self.head_dim), attention_dropout=0.)

    def _forward(self, 
                 hidden_states: torch.Tensor, index_ids, cu_seqlens):
        if not self.training:
            self.hidden_states = hidden_states
        else:
            self.hidden_states = None
        assert hidden_states.ndim == 2, f"hidden_states.shape must be (T, H), but got {hidden_states.shape}"

        _hidden_states = self.input_layernorm(hidden_states)
        query, key, value = self.self_attn["q_proj"](_hidden_states), self.self_attn["k_proj"](
            _hidden_states), self.self_attn["v_proj"](_hidden_states)
        query, key, value = rearrange(query, "t (h d) -> t h d", d=self.head_dim), \
            rearrange(key, "t (h d) -> t h d", d=self.head_dim), \
            rearrange(value, "t (h d) -> t h d", d=self.head_dim)
        query, key = self.self_attn["rotary_emb"](query, key, index_ids)
        
        if self.config.pe_config['1d']:
            value = torch.cat([value, torch.zeros_like(value, device=value.device, dtype=value.dtype)], dim=-1)

        if self.num_key_value_groups > 1:
            key = torch.repeat_interleave(key, dim=2, repeats=self.num_key_value_groups)
            value = torch.repeat_interleave(value, dim=2, repeats=self.num_key_value_groups)
        
        assert FlashAttention is not None, \
            "Detected flash_attn is not installed. See https://github.com/HazyResearch/flash-attention"
        
        qkv = torch.stack([query, key, value], dim=1)
        output = self.flash_attn(qkv, cu_seqlens=cu_seqlens, max_seqlen=int(index_ids.max()), causal=True)
        
        if self.config.pe_config['1d']:
            output = output[..., :self.head_dim]
        
        output = rearrange(output, "t h d -> t (h d)")

        output = F.dropout(output, p=self.config.dropout, training=self.training)
        hidden_states = hidden_states + self.self_attn["o_proj"](output)
        _hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + F.dropout(self.mlp["down_proj"](F.silu(self.mlp["gate_proj"](
            _hidden_states)) * self.mlp["up_proj"](_hidden_states)), p=self.config.dropout, training=self.training)
        return hidden_states

    def forward(self, inputs: dict):
        if self.config.checkpointing and self.training:
            inputs["hidden_states"] = torch.utils.checkpoint.checkpoint(
                self._forward,
                inputs["hidden_states"],
                inputs["index_ids"],
                inputs["cu_seqlens"]
            )
        else:
            inputs["hidden_states"] = self._forward(**inputs)
        return inputs


class LlamaForCausalLM(CollieModelForCausalLM):
    
    def __init__(self, config: CollieConfig) -> None:
        super().__init__(config)
        self.embed_tokens = tensor_parallel.VocabParallelEmbedding(
            self.collie_config.vocab_size,
            self.collie_config.hidden_size,
            perform_initialization=False
        )
        self.layers = nn.ModuleList(
            [LlamaLayer(self.collie_config) for _ in range(self.collie_config.num_hidden_layers)])
        self.norm = RMSNormalize(
            dim=self.collie_config.hidden_size,
            eps=self.collie_config.rms_norm_eps
        )
        self.lm_head = ColumnParallelLinearWithoutBias(
            self.collie_config.hidden_size,
            self.collie_config.vocab_size,
            bias=False
        )
        # GenerationMixin 需要的额外参数
        self.config.is_decoder=True
        if config.model_config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.main_input_name = "input_ids"

    def forward(self, 
                input_ids: Optional[torch.Tensor] = None, 
                attention_mask: Optional[torch.Tensor] = None, 
                past_key_values: Optional[Tuple[torch.Tensor]] = None,
                **kwargs):
        
        assert input_ids.ndim == 2
        batch, seqlen = input_ids.shape
        
        if attention_mask is None:
            attention_mask = (input_ids != 0).int()
        
        seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
        indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))

        input_ids = torch.gather(input_ids.flatten(), 0, indices).reshape(-1,)
        
        # input_ids, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(input_ids, attention_mask.bool())
        
        index_ids = torch.arange(seqlen, device='cuda').reshape((1, -1)) * attention_mask
        index_ids = torch.gather(index_ids.flatten(), 0, indices).reshape(-1,)

        inputs = {"hidden_states": self.embed_tokens(input_ids), "index_ids": index_ids, "cu_seqlens": cu_seqlens}
        
        all_hidden_states = ()
        for layer in self.layers:
            
            all_hidden_states += (inputs["hidden_states"],)
            inputs.update(layer(inputs))
            
        inputs["hidden_states"] = pad_input(inputs["hidden_states"], indices, batch, seqlen)
        
        inputs["hidden_states"] = self.norm(inputs["hidden_states"])
        all_hidden_states += (inputs["hidden_states"], )
        inputs["logits"] = self.lm_head(inputs["hidden_states"])
        return CausalLMOutputWithPast(
            loss=None,
            logits=inputs["logits"],
            past_key_values=self._get_past_key_values(self.layers),
            hidden_states=all_hidden_states,
            attentions=None
        )

    def clean(self):
        self._clean_hidden_states([*self.layers, self.lm_head])
        self._clean_past_key_values(self.layers)
        self._set_use_cache(self.layers, False)
        
    def set_cache(self, use_cache, past_key_values):
        self._set_use_cache(self.layers, use_cache)
        self._set_past_key_values(self.layers, past_key_values)

    @classmethod
    def pipeline_layers(cls, config: CollieConfig):
        """
        Get layers of pipeline.

        :return: list
        """
        if isinstance(config, str):
            config = CollieConfig.from_pretrained(config)
        if config.model_config.tie_word_embeddings:
            return [TiedLayerSpec(
                "embed_tokens",
                dict_as_params(input_keys="input_ids", output_keys="hidden_states"),
                tensor_parallel.VocabParallelEmbedding,
                config.vocab_size,
                config.hidden_size),
                *[LayerSpec(LlamaLayer, config)
                for _ in range(config.num_hidden_layers)],
                LayerSpec(dict_as_params(input_keys="hidden_states", output_keys="hidden_states"), RMSNormalize,
                        dim=config.hidden_size,
                        eps=config.rms_norm_eps),
                TiedLayerSpec(
                "embed_tokens",
                dict_as_params(input_keys="hidden_states", output_keys="logits"),
                ColumnParallelLMHead,
                config.hidden_size,
                config.vocab_size,
                bias=False)
            ]
        else:
            return [LayerSpec(
                dict_as_params(input_keys="input_ids", output_keys="hidden_states"),
                tensor_parallel.VocabParallelEmbedding,
                config.vocab_size,
                config.hidden_size),
                *[LayerSpec(LlamaLayer, config)
                for _ in range(config.num_hidden_layers)],
                LayerSpec(dict_as_params(input_keys="hidden_states", output_keys="hidden_states"), RMSNormalize,
                        dim=config.hidden_size,
                        eps=config.rms_norm_eps),
                LayerSpec(
                dict_as_params(input_keys="hidden_states", output_keys="logits"),
                ColumnParallelLMHead,
                config.hidden_size,
                config.vocab_size,
                bias=False)
            ]

    @staticmethod
    def load_parallel_state_dict(path: str, config: Union[CollieConfig, str],
                                 process_exclusion: bool = False, **kwargs):...
    @staticmethod
    def load_parallel_state_dict(path: str,
                                 config: Union[CollieConfig, str],
                                 process_exclusion: bool = False,
                                 protocol: str = 'file',
                                 format: str = 'hf', **kwargs):
        """
        Load state_dict from ``path``.

        The format of pretrained model should be the same as that of
        `huggingface`.

        :return: state_dict. Note that the state_dict should be processed
            properly to match the current rank.
        """
        assert format in ["hf", "meta"], "Only support hf and meta format"
        if isinstance(config, str):
            config = CollieConfig.from_pretrained(config)
        io_driver = IODriver.from_protocol(protocol)
        if not io_driver.exists(path):
            raise FileNotFoundError(f"folder {path} not found.")
        state_dict = OrderedDict()
        weights = []
        parts = None
        # 如果开启了进程互斥，那么每个进程都会显示进度条，否则只显示 RANK0 的
        hide_progress = not process_exclusion and int(os.environ.get("RANK", "0")) != 0
        if dist.is_initialized() and process_exclusion:
            # 如果启动了进程互斥，则要进行 dist.get_world_size() 次循环
            rank_order = range(dist.get_world_size())
        else:
            # 不开启只进行一次循环
            rank_order = range(1)
        for rank in rank_order:
            # 如果开启了进程互斥，那么只有对应 RANK 的能进入循环；不开启进程互斥的话就都可以进
            if int(os.environ.get("RANK", "0")) == rank or not process_exclusion:
                # PP 分层的方法保存在了 os.environ["COLLIE_PP_PARTS"], 格式类似于 [0, 17, 35], 左闭右开
                if env.is_pipeline:
                    # 保存的是 json 格式
                    parts = env.pipeline_parts
                if format == "hf":
                    # 如果存在 pytorch_model.bin.index.json 文件的话，此时不同的 pp 进程可以按需加载自己需要的权重
                    if io_driver.exists(os.path.join(path, "pytorch_model.bin.index.json")) and "COLLIE_PP_PARTS" in os.environ.keys():
                        weight_map = json.loads(io_driver.load(os.path.join(path, "pytorch_model.bin.index.json"), mode="r"))["weight_map"]
                        # layers 表示自己需要的层
                        layers = list(range(parts[int(os.environ["COLLIE_PP_RANK"])], parts[int(os.environ["COLLIE_PP_RANK"]) + 1]))
                        # 筛选出形似 model.layers.0 这样的层。包含两个条件：1. 有数字的层；2. 数字加一要在 layers 里面（因为最开始还有个 embedding 占一层）
                        weights.extend([value for key, value in weight_map.items() \
                            if len(key.split(".")) > 2 \
                                and key.split(".")[2].isdigit() \
                                    and (int(key.split(".")[2]) + 1) in layers])
                        # 去重
                        weights = list(set(weights))
                        # 继续筛选，如果有 0 层，那么就要加载 embedding；如果有最后一层，那么就要加载 lm_head；如果有倒数第二层，那么就要加载 norm
                        if 0 in layers:
                            weights.append(weight_map["model.embed_tokens.weight"])
                        if max(parts) - 1 in layers:
                            weights.append(weight_map["lm_head.weight"])
                        if max(parts) - 2 in layers:
                            weights.append(weight_map["model.norm.weight"])
                    else:
                        # 如果没有 pytorch_model.bin.index.json 文件的话，那么就加载所有的权重
                        weights = [weight for weight in io_driver.list(path) if weight.endswith(".bin")]
                    with progress(weights, desc="Loading state dict", total=len(weights), disable=hide_progress) as pbar:
                        for weight in pbar:
                            part_state_dict = io_driver.load(os.path.join(path, weight), mode="rb")
                            for key in list(part_state_dict.keys()):
                                # 对 q_proj.weight 和 k_proj.weight 进行 reshape
                                if key.endswith("q_proj.weight") or key.endswith("k_proj.weight"):
                                    part_state_dict[key] = rearrange(
                                        part_state_dict[key],
                                        "(h two t) d -> h two t d",
                                        h=config.num_attention_heads,
                                        two=2).transpose(1, 2).reshape(
                                            config.hidden_size,
                                            config.hidden_size)
                                part_state_dict[key.replace("model.", "")] = part_state_dict.pop(key)
                            state_dict.update(part_state_dict)
                            del part_state_dict
                elif format == "meta":
                    # 根据 meta 中的 params.json 更新一下用户配置
                    if io_driver.exists(os.path.join(path, "params.json")):
                        params = json.loads(io_driver.load(os.path.join(path, "params.json"), mode="r"))
                        for key, value in {
                            "hidden_size": params["dim"],
                            "intermediate_size": params["multiple_of"] * ((int(2 * 4 * config.hidden_size / 3) + params["multiple_of"] - 1) // params["multiple_of"]),
                            "num_hidden_layers": params["n_layers"],
                            "num_attention_heads": params["n_heads"],
                            "rms_norm_eps": params["norm_eps"]
                        }.items():
                            setattr(config, key, value)
                    # 权重全部加载
                    weights = [weight for weight in io_driver.list(path) if (weight.endswith(".pt") or weight.endswith(".pth"))]
                    # 因为 meta 的权重默认 按照张量并行分割，cat 的时候存在顺序问题，所以先排序一下
                    weights = sorted(weights, key=lambda x: int(x.split(".")[1]))
                    with progress(weights, desc="Loading state dict", total=len(weights), disable=hide_progress) as pbar:
                        for weight in pbar:
                            part_state_dict = io_driver.load(os.path.join(path, weight), mode="rb")
                            for key in list(part_state_dict.keys()):
                                # if key.startswith("layers"):
                                #     layer = int(key.split(".")[1])
                                raw_key = key
                                key = key.replace("attention", "self_attn")
                                key = key.replace("wo", "o_proj")
                                key = key.replace("wq", "q_proj")
                                key = key.replace("wk", "k_proj")
                                key = key.replace("wv", "v_proj")
                                key = key.replace("feed_forward", "mlp")
                                key = key.replace("w1", "gate_proj")
                                key = key.replace("w2", "down_proj")
                                key = key.replace("w3", "up_proj")
                                key = key.replace("self_attn_norm", "input_layernorm")
                                key = key.replace("ffn_norm", "post_attention_layernorm")
                                key = key.replace("tok_embeddings", "embed_tokens")
                                key = key.replace("output", "lm_head")
                                # 按照 hf 的格式更新字典
                                part_state_dict[key] = part_state_dict.pop(raw_key)
                            for key in list(part_state_dict.keys()):
                                if not key in state_dict.keys():
                                    state_dict[key] = part_state_dict[key]
                                else:
                                    # 组装一下
                                    if key.endswith("q_proj.weight") \
                                        or key.endswith("k_proj.weight") \
                                            or key.endswith("v_proj.weight") \
                                                or key.endswith("gate_proj.weight") \
                                                    or key.endswith("up_proj.weight") \
                                                        or key.endswith("lm_head.weight"):
                                                            state_dict[key] = torch.cat((state_dict[key], part_state_dict[key]), dim=0)
                                    if key.endswith("o_proj.weight") \
                                        or key.endswith("down_proj.weight") \
                                            or key.endswith("embed_tokens.weight"):
                                                state_dict[key] = torch.cat((state_dict[key], part_state_dict[key]), dim=1)
                            del part_state_dict
                if parts is not None:
                    # 这一步是 pp 的复筛
                    layers = list(range(parts[int(os.environ["COLLIE_PP_RANK"])], parts[int(os.environ["COLLIE_PP_RANK"]) + 1]))
                    for key in list(state_dict.keys()):
                        if key.startswith("layers"):
                            layer = int(key.split(".")[1])
                            if layer + 1 in layers:
                                state_dict[key.replace(f"layers.{layer}", f"{layer + 1}")] = state_dict.pop(key)
                            else:
                                # 形似 model.layers.0 这样的层，筛选掉数字加一不在 layers 里面得
                                state_dict.pop(key)
                        if key.endswith("embed_tokens.weight"):
                            if 0 in layers:
                                if config.model_config.tie_word_embeddings:
                                    state_dict["tied_modules.embed_tokens.weight"] = state_dict.pop(key)
                                else:
                                    state_dict["0.weight"] = state_dict.pop(key)
                            else:
                                state_dict.pop(key)
                        if key == "norm.weight":
                            if max(parts) - 2 in layers:
                                state_dict[f"{max(parts) - 2}.weight"] = state_dict.pop(key)
                            else:
                                state_dict.pop(key)
                        if key.endswith("lm_head.weight"):
                            if max(parts) - 1 in layers:
                                if config.model_config.tie_word_embeddings:
                                    state_dict["tied_modules.embed_tokens.weight"] = state_dict.pop(key)
                                else:
                                    state_dict[f"{max(parts) - 1}.weight"] = state_dict.pop(key)
                            else:
                                state_dict.pop(key)
                # 根据用户配置的新的 tp size 进行分割
                for key in list(state_dict.keys()):
                    filte_list = ["q_proj.weight", "q_proj.weight", "k_proj.weight", "v_proj.weight", "gate_proj.weight", "up_proj.weight", "embed_tokens.weight", "lm_head.weight"]
                    need_split = any([key.endswith(filte) for filte in filte_list])
                    if env.pp_size > 1:
                        # embedding 层和 lm_head 都需要切
                        need_split = need_split or int(key.split(".")[0]) == max(parts) - 1
                        need_split = need_split or int(key.split(".")[0]) == min(parts)
                    if need_split:
                        tensor = list(torch.chunk(state_dict[key], config.tp_size, dim=0))[int(os.environ.get("COLLIE_TP_RANK", "0"))].detach().clone()
                        del state_dict[key]
                        if process_exclusion:
                            # CPU 内存回收（速度很慢）
                            gc.collect()
                        state_dict[key] = tensor
                    elif key.endswith("o_proj.weight") \
                        or key.endswith("down_proj.weight"):
                            tensor = list(torch.chunk(state_dict[key], config.tp_size, dim=1))[int(os.environ.get("COLLIE_TP_RANK", "0"))].detach().clone()
                            del state_dict[key]
                            if process_exclusion:
                                # CPU 内存回收（速度很慢）
                                gc.collect()
                            state_dict[key] = tensor
            if dist.is_initialized() and process_exclusion:
                # 如果选择了进程互斥，那么本次循环中不需要加载权重的进程需等待
                dist.barrier()
        return state_dict

    @staticmethod
    def save_parallel_state_dict(state_dict: dict, path: str,
                                 config: CollieConfig,
                                 process_exclusion: bool = False, **kwargs):...
    @staticmethod
    def save_parallel_state_dict(state_dict: dict,
                                 path: str,
                                 config: CollieConfig,
                                 process_exclusion: bool = False,
                                 protocol: str = 'file'):
        """
        Save state_dict to ``path``.

        The format of saved state dict should be the same as that of
        `huggingface`.
        """
        io_driver = IODriver.from_protocol(protocol)
        def reshape_wq_wk(w: torch.Tensor):
            return w.view(config.num_attention_heads,
                          config.hidden_size // config.num_attention_heads // 2,
                          2,
                          config.hidden_size).transpose(1, 2).reshape(config.hidden_size,
                                                                    config.hidden_size)
        # gather to tp rank 0
        if env.is_pipeline:
            layers = env.pipeline_layers_idx
            parts = env.pipeline_parts
            for key in list(state_dict.keys()):
                if key == "tied_modules.embed_tokens.weight":
                    if 0 in layers:
                        state_dict["model.embed_tokens.weight"] = state_dict.pop(key)
                    elif max(layers) - 1 in layers:
                        state_dict["lm_head.weight"] = state_dict.pop(key)
                else:
                    layer = int(key.split(".")[0])
                    if layer == max(parts) - 2:
                        state_dict[key.replace(f"{layer}.", "model.norm.")] = state_dict.pop(key)
                    else:
                        state_dict[key.replace(f"{layer}.", f"model.layers.{layer - 1}.")] = state_dict.pop(key)
        if dist.is_initialized() and process_exclusion:
            # 如果启动了进程互斥，则要进行 pp_size 次循环
            rank_order = range(config.pp_size)
        else:
            # 不开启只进行一次循环
            rank_order = range(1)
        dst = parallel_state.get_tensor_model_parallel_src_rank()
        with progress(rank_order, desc="Saving model", disable=int(os.environ.get("RANK", "0")) != 0) as pbar:
            for rank in pbar:
                if env.dp_rank == 0 \
                    and (env.pp_rank == rank
                         or not process_exclusion):
                    for key in sorted(list(state_dict.keys())):
                        tensor_list = None
                        if env.tp_size > 1:
                            if env.tp_rank == 0:
                                tensor_list = [torch.zeros_like(state_dict[key]).to(state_dict[key].dtype).cuda() for _ in range(config.tp_size)]
                            dist.gather(state_dict[key].cuda(), dst=dst, gather_list=tensor_list, group=env.tp_group)
                            if env.tp_rank == 0:
                                filte_list = ["q_proj.weight", "q_proj.weight", "k_proj.weight", "v_proj.weight", "gate_proj.weight", "up_proj.weight", "embed_tokens.weight", "lm_head.weight"]
                                need_split = any([key.endswith(filte) for filte in filte_list])
                                if env.pp_size > 1:
                                    # embedding 层和 lm_head 都需要切
                                    need_split = need_split or int(key.split(".")[0]) == max(parts) - 1
                                    need_split = need_split or int(key.split(".")[0]) == min(parts)

                                if need_split:
                                    state_dict[key] = concat_tensor(tensor_list, dim=0)
                                    if process_exclusion:
                                        # CPU 内存回收（速度很慢）
                                        gc.collect()
                                                            
                                elif key.endswith("o_proj.weight") \
                                    or key.endswith("down_proj.weight"):
                                        state_dict[key] = concat_tensor(tensor_list, dim=1)
                                        if process_exclusion:
                                            # CPU 内存回收（速度很慢）
                                            gc.collect()
                        if key.endswith("q_proj.weight")  or key.endswith("k_proj.weight"):
                            state_dict[key] = reshape_wq_wk(state_dict[key])
                        if not key.startswith("lm_head.weight"):
                            state_dict[f"model.{key}"] = state_dict.pop(key)
                    if env.tp_rank == 0:
                        # Save gathered weights
                        if env.is_pipeline:
                            ckpt_name = f"pytorch_model-{env.pp_rank+1:05d}-of-{config.pp_size:05d}.bin"
                            total_size = 0
                            weight_map = {}
                            for name, weight in state_dict.items():
                                weight_size = weight.numel() * dtype_byte_size(weight.dtype)
                                weight_map[name] = ckpt_name
                                total_size += weight_size
                            index_dict = dict(total_size=total_size, weight_map=weight_map)
                            tmp_index_file = os.path.join(path, "_tmp_index_{}.json")
                            io_driver.save(
                                json.dumps(index_dict), tmp_index_file.format(env.pp_rank)
                            )
                        else:
                            ckpt_name = f"pytorch_model.bin"
                        ckpt_path = os.path.join(path, ckpt_name)
                        io_driver.save(state_dict, ckpt_path)
                if dist.is_initialized() and process_exclusion:
                    dist.barrier()
        dist.barrier()
        if env.rank == 0:
            config.save_pretrained(path)
        if env.rank == 0 and env.is_pipeline:
            # merge
            tmp_index_files = [tmp_index_file.format(i) for i in range(config.pp_size)]
            total_size = 0
            weight_map = {}
            for _file in tmp_index_files:
                _index_dict = json.loads(io_driver.load(_file, mode="r"))
                total_size += _index_dict["total_size"]
                weight_map.update(_index_dict["weight_map"])
                io_driver.delete(_file)
            merged_dict = {
                "metadata": {"total_size": total_size},
                "weight_map": weight_map
            }
            io_driver.save(
                json.dumps(merged_dict, indent=2, sort_keys=True) + "\n",
                os.path.join(path, "pytorch_model.bin.index.json")
            )
