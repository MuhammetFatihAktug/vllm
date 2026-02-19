# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SigLIP2 embedding model for vLLM.

Adapted from siglip.py for SigLIP2's NaFlex (Native resolution Flexible)
architecture. Key differences from SigLIP v1:
- Pre-patchified input: pixel_values are (batch, max_patches, patch_dim)
- Resizable 2D positional embeddings via bilinear interpolation
- Attention mask for padding patches
- MultiheadAttentionPoolingHead with mask support
"""

from collections.abc import Callable, Iterable, Mapping
from functools import cached_property
from typing import Annotated, Literal

import torch
import torch.nn.functional as F
from torch import nn
from transformers import BatchFeature

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.pooler import DispatchPooler
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalInputs,
    MultiModalKwargsItems,
    MultiModalUUIDDict,
)
from vllm.multimodal.parse import (
    ImageProcessorItems,
    ImageSize,
    MultiModalDataItems,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptIndexTargets,
    PromptReplacement,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsQuant
from .interfaces_base import default_pooling_type
from .utils import AutoWeightsLoader, maybe_prefix
from .vision import (
    VisionEncoderInfo,
    is_vit_use_data_parallel,
)

# Try to import SigLIP2 configs from transformers
try:
    from transformers import (
        Siglip2Config,
        Siglip2Processor,
    )
    from transformers.models.siglip2.configuration_siglip2 import (
        Siglip2TextConfig,
        Siglip2VisionConfig,
    )
except ImportError:
    raise ImportError(
        "SigLIP2 requires transformers >= 4.49.0. "
        "Please upgrade: pip install -U transformers"
    )

from .siglip import _get_vision_feature_select_strategy
from .vision import (
    VisionFeatureSelectStrategy,
    resolve_visual_encoder_outputs,
)


class Siglip2ImagePixelInputs(TensorSchema):
    """
    SigLIP2 NaFlex pre-patchified image inputs.

    Dimensions:
        - bn: Batch size * number of images
        - max_patches: Maximum number of patches (from config.num_patches)
        - patch_dim: patch_size * patch_size * num_channels
    """

    type: Literal["pixel_values"]
    data: Annotated[torch.Tensor, TensorShape("bn", "max_patches", "patch_dim")]


# ==================== Processing / Info ====================


class Siglip2EncoderInfo(VisionEncoderInfo[Siglip2VisionConfig]):
    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        # The attention pooling head reduces all patches to a single token.
        return 1

    def get_image_size(self) -> int:
        # SigLIP2 doesn't have a fixed image_size; use patch_size * sqrt(num_patches)
        # as a representative size for dummy input generation
        grid_len = self.get_patch_grid_length()
        return grid_len * self.vision_config.patch_size

    def get_patch_size(self) -> int:
        return self.vision_config.patch_size

    def get_patch_grid_length(self) -> int:
        return int(self.vision_config.num_patches ** 0.5)


class Siglip2ProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Siglip2Config)

    def get_vision_encoder_info(self):
        return Siglip2EncoderInfo(self.get_hf_config())

    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(Siglip2Processor, **kwargs)

    def get_image_processor(self, **kwargs: object):
        return self.get_hf_processor(**kwargs).image_processor

    def get_max_num_patches(self, **kwargs: object) -> int:
        """Get effective max_num_patches from kwargs or config default."""
        if "max_num_patches" in kwargs:
            return int(kwargs["max_num_patches"])
        return self.get_image_processor(**kwargs).max_num_patches

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": 1}

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        max_num_patches: int | None = None,
    ) -> int:
        # Attention pooling head reduces all patches to 1 token
        return 1

    def get_image_size_with_most_features(
        self, max_num_patches: int | None = None,
    ) -> ImageSize:
        if max_num_patches is None:
            max_num_patches = self.get_max_num_patches()
        patch_size = self.get_vision_encoder_info().get_patch_size()
        grid_len = int(max_num_patches ** 0.5)
        size = grid_len * patch_size
        return ImageSize(width=size, height=size)

    def get_max_image_tokens(self) -> int:
        # Attention pooling head reduces all patches to 1 token
        return 1


class Siglip2DummyInputsBuilder(
    BaseDummyInputsBuilder[Siglip2ProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        target_width, target_height = (
            self.info.get_image_size_with_most_features()
        )
        image_overrides = mm_options.get("image") if mm_options else None

        return {
            "image": self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=num_images,
                overrides=image_overrides,
            )
        }


class Siglip2MultiModalProcessor(
    BaseMultiModalProcessor[Siglip2ProcessingInfo]
):
    @cached_property
    def image_token_id(self) -> int:
        tokenizer = self.info.get_tokenizer()
        dummy_token_id = next(
            token_id
            for token_id in range(tokenizer.vocab_size)
            if token_id not in tokenizer.all_special_ids
        )
        return dummy_token_id

    def apply(
        self,
        prompt: str | list[int],
        mm_data: MultiModalDataDict,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object] | None = None,
        *,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> MultiModalInputs:
        if prompt and mm_data:
            raise ValueError(
                "Siglip2 accepts text-only or image-only inputs, not both! "
                "Image-only inputs means passing an image with an empty text "
                "prompt."
            )

        if mm_data:
            tokenization_kwargs = {
                **(tokenization_kwargs or {}),
                "add_special_tokens": False,
            }

        return super().apply(
            prompt=prompt,
            mm_data=mm_data,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            tokenization_kwargs=tokenization_kwargs,
            mm_uuids=mm_uuids,
        )

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return False

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            pixel_attention_mask=MultiModalFieldConfig.batched("image"),
            spatial_shapes=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> list[PromptUpdate]:
        image_token_id = self.image_token_id
        max_num_patches = hf_processor_mm_kwargs.get(
            "max_num_patches", None
        )

        def get_replacement(item_idx: int):
            images = mm_items.get_items("image", ImageProcessorItems)
            image_size = images.get_image_size(item_idx)
            num_image_tokens = self.info.get_num_image_tokens(
                image_width=image_size.width,
                image_height=image_size.height,
                max_num_patches=max_num_patches,
            )
            return [image_token_id] * num_image_tokens

        return [
            PromptReplacement(
                modality="image",
                target=PromptIndexTargets.start(),
                replacement=get_replacement,
            ),
        ]


# ==================== Vision Encoder ====================


class Siglip2VisionEmbeddings(nn.Module):
    """SigLIP2 NaFlex vision embeddings with resizable positional embeddings."""

    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size
        self.num_patches = config.num_patches
        self.position_embedding_size = int(self.num_patches ** 0.5)

        # SigLIP2 uses Linear instead of Conv2d (input is pre-patchified)
        self.patch_embedding = nn.Linear(
            in_features=config.num_channels * self.patch_size * self.patch_size,
            out_features=self.embed_dim,
        )

        self.position_embedding = nn.Embedding(
            self.num_patches, self.embed_dim
        )

    @staticmethod
    def resize_positional_embeddings(
        positional_embeddings: torch.Tensor,
        spatial_shapes: torch.LongTensor,
        max_length: int,
    ) -> torch.Tensor:
        """Resize positional embeddings via bilinear interpolation.

        Args:
            positional_embeddings: (height, width, embed_dim)
            spatial_shapes: (batch_size, 2) with (h_patches, w_patches)
            max_length: pad to this length
        Returns:
            (batch_size, max_length, embed_dim)
        """
        batch_size = spatial_shapes.shape[0]
        embed_dim = positional_embeddings.shape[-1]
        source_dtype = positional_embeddings.dtype

        resulted_positional_embeddings = torch.empty(
            (batch_size, max_length, embed_dim),
            device=positional_embeddings.device,
            dtype=source_dtype,
        )

        # (height, width, embed_dim) -> (1, embed_dim, height, width)
        positional_embeddings = positional_embeddings.permute(
            2, 0, 1
        ).unsqueeze(0)

        if positional_embeddings.device.type == "cpu":
            positional_embeddings = positional_embeddings.to(torch.float32)

        for i in range(batch_size):
            height, width = spatial_shapes[i]
            resized_embeddings = F.interpolate(
                positional_embeddings,
                size=(int(height), int(width)),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            resized_embeddings = resized_embeddings.reshape(
                embed_dim, int(height) * int(width)
            ).transpose(0, 1)
            resized_embeddings = resized_embeddings.to(source_dtype)

            num_valid = int(height) * int(width)
            resulted_positional_embeddings[i, :num_valid] = resized_embeddings
            # Pad with first embedding (following HF implementation)
            resulted_positional_embeddings[i, num_valid:] = (
                resized_embeddings[0]
            )

        return resulted_positional_embeddings

    def forward(
        self,
        pixel_values: torch.Tensor,
        spatial_shapes: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Args:
            pixel_values: (batch, max_num_patches, patch_dim)
            spatial_shapes: (batch, 2) with (h_patches, w_patches)
        """
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(
            pixel_values.to(dtype=target_dtype)
        )

        positional_embeddings = self.position_embedding.weight.reshape(
            self.position_embedding_size, self.position_embedding_size, -1
        )
        resized_positional_embeddings = self.resize_positional_embeddings(
            positional_embeddings,
            spatial_shapes,
            max_length=pixel_values.shape[1],
        )

        embeddings = patch_embeds + resized_positional_embeddings
        return embeddings


class Siglip2Attention(nn.Module):
    """SigLIP2 vision attention using PyTorch SDPA.

    Uses F.scaled_dot_product_attention directly instead of vLLM's
    EncoderOnlyAttention, because embed_multimodal runs outside the
    forward context required by the unified attention kernel.
    """

    def __init__(
        self,
        config: Siglip2VisionConfig | Siglip2TextConfig,
        quant_config: QuantizationConfig | None = None,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads "
                f"(got `embed_dim`: {self.embed_dim} and "
                f"`num_heads`: {self.num_heads})."
            )

        self.scale = self.head_dim ** -0.5

        use_data_parallel = is_vit_use_data_parallel()
        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.embed_dim,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
            disable_tp=use_data_parallel,
        )

        self.out_proj = RowParallelLinear(
            input_size=self.embed_dim,
            output_size=self.embed_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
            disable_tp=use_data_parallel,
        )

        self.tp_size = (
            1
            if use_data_parallel
            else get_tensor_model_parallel_world_size()
        )
        self.num_heads_per_partition = divide(self.num_heads, self.tp_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        # Handle both 2D (total_tokens, hidden) from text path
        # and 3D (batch, seq_len, hidden) from vision path
        is_2d = hidden_states.ndim == 2
        if is_2d:
            hidden_states = hidden_states.unsqueeze(0)

        batch_size, seq_len, _ = hidden_states.shape

        qkv_states, _ = self.qkv_proj(hidden_states)
        query_states, key_states, value_states = qkv_states.chunk(3, dim=-1)

        # Reshape to (batch, num_heads, seq_len, head_dim)
        query_states = query_states.view(
            batch_size, seq_len, self.num_heads_per_partition, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, seq_len, self.num_heads_per_partition, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, seq_len, self.num_heads_per_partition, self.head_dim
        ).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, scale=self.scale
        )

        # Reshape back to (batch, seq_len, embed_dim)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, -1)
        )

        attn_output, _ = self.out_proj(attn_output)

        if is_2d:
            attn_output = attn_output.squeeze(0)

        return attn_output, None


class Siglip2MLP(nn.Module):
    def __init__(
        self,
        config: Siglip2VisionConfig | Siglip2TextConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        use_data_parallel = is_vit_use_data_parallel()
        self.activation_fn = get_act_fn(config.hidden_act)

        if quant_config and quant_config.get_name() in [
            "bitsandbytes",
            "torchao",
        ]:
            quantizable = True
        else:
            quantizable = (
                config.hidden_size % 64 == 0
                and config.intermediate_size % 64 == 0
            )

        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            quant_config=quant_config if quantizable else None,
            prefix=f"{prefix}.fc1",
            disable_tp=use_data_parallel,
        )
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            quant_config=quant_config if quantizable else None,
            prefix=f"{prefix}.fc2",
            disable_tp=use_data_parallel,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


class Siglip2EncoderLayer(nn.Module):
    def __init__(
        self,
        config: Siglip2VisionConfig | Siglip2TextConfig,
        quant_config: QuantizationConfig | None = None,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.embed_dim = config.hidden_size

        self.self_attn = Siglip2Attention(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.layer_norm1 = nn.LayerNorm(
            self.embed_dim, eps=config.layer_norm_eps
        )
        self.mlp = Siglip2MLP(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.layer_norm2 = nn.LayerNorm(
            self.embed_dim, eps=config.layer_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        hidden_states += residual

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states += residual

        return hidden_states, None


class Siglip2Encoder(nn.Module):
    def __init__(
        self,
        config: Siglip2VisionConfig | Siglip2TextConfig,
        quant_config: QuantizationConfig | None = None,
        num_hidden_layers_override: int | None = None,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        if num_hidden_layers_override is None:
            num_hidden_layers = config.num_hidden_layers
        else:
            num_hidden_layers = num_hidden_layers_override

        self.layers = nn.ModuleList(
            [
                Siglip2EncoderLayer(
                    config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{layer_idx}",
                )
                for layer_idx in range(num_hidden_layers)
            ]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        return_all_hidden_states: bool,
    ) -> torch.Tensor | list[torch.Tensor]:
        hidden_states_pool = [inputs_embeds]
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            hidden_states, _ = encoder_layer(hidden_states)
            if return_all_hidden_states:
                hidden_states_pool.append(hidden_states)

        if return_all_hidden_states:
            return hidden_states_pool
        return hidden_states


class Siglip2MultiheadAttentionPoolingHead(nn.Module):
    """Multihead Attention Pooling with mask support for NaFlex padding."""

    def __init__(
        self,
        config: Siglip2VisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = torch.nn.MultiheadAttention(
            config.hidden_size,
            config.num_attention_heads,
            batch_first=True,
        )
        self.layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.mlp = Siglip2MLP(
            config=config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.num_heads = config.num_attention_heads

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = hidden_state.size(0)
        probe = self.probe.expand(batch_size, -1, -1)

        if attention_mask is not None:
            # Convert (batch, seq) mask to attention mask for MultiheadAttention
            # MultiheadAttention expects: (batch*num_heads, tgt_len, src_len)
            target_len = probe.shape[1]  # 1
            source_len = hidden_state.shape[1]

            # Create 4D mask: (batch, 1, 1, src_len)
            # where 0 = attend, -inf = ignore
            expanded_mask = attention_mask[:, None, None, :].to(
                dtype=hidden_state.dtype
            )
            expanded_mask = (1.0 - expanded_mask) * torch.finfo(
                hidden_state.dtype
            ).min

            # Expand to (batch, num_heads, target_len, source_len)
            expanded_mask = expanded_mask.expand(
                batch_size, self.num_heads, target_len, source_len
            )
            # Reshape to (batch * num_heads, target_len, source_len)
            attn_mask = expanded_mask.reshape(
                batch_size * self.num_heads, target_len, source_len
            )
        else:
            attn_mask = None

        hidden_state = self.attention(
            probe, hidden_state, hidden_state, attn_mask=attn_mask
        )[0]

        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = self.mlp(hidden_state)
        hidden_state += residual

        # Return (batch, 1, hidden_size) to match resolve_visual_encoder_outputs
        return hidden_state


class Siglip2VisionTransformer(nn.Module):
    def __init__(
        self,
        config: Siglip2VisionConfig,
        quant_config: QuantizationConfig | None = None,
        *,
        num_hidden_layers_override: int | None = None,
        require_post_norm: bool | None = None,
        prefix: str = "",
        use_head: bool | None = False,
    ) -> None:
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = Siglip2VisionEmbeddings(config)

        self.encoder = Siglip2Encoder(
            config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers_override,
            prefix=f"{prefix}.encoder",
        )

        num_hidden_layers = config.num_hidden_layers
        if len(self.encoder.layers) > config.num_hidden_layers:
            raise ValueError(
                f"The original encoder only has {num_hidden_layers} "
                f"layers, but you requested {len(self.encoder.layers)} layers."
            )

        if require_post_norm is None:
            require_post_norm = (
                len(self.encoder.layers) == num_hidden_layers
            )

        if require_post_norm:
            self.post_layernorm = nn.LayerNorm(
                embed_dim, eps=config.layer_norm_eps
            )
        else:
            self.post_layernorm = None

        if isinstance(use_head, bool):
            self.use_head = use_head
        else:
            self.use_head = (
                True
                if not hasattr(config, "vision_use_head")
                else config.vision_use_head
            )

        self.head = (
            Siglip2MultiheadAttentionPoolingHead(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.head",
            )
            if self.use_head
            else None
        )

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        pixel_values: torch.Tensor,
        spatial_shapes: torch.LongTensor,
        pixel_attention_mask: torch.Tensor | None = None,
        *,
        select_layers: list[int] | None = None,
        feature_select_strategy: VisionFeatureSelectStrategy | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values, spatial_shapes)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            return_all_hidden_states=select_layers is not None,
        )

        # Apply post_layernorm and head as last_hs_proc
        def last_hs_proc(encoder_outputs: torch.Tensor) -> torch.Tensor:
            if self.post_layernorm is not None:
                encoder_outputs = self.post_layernorm(encoder_outputs)
            if self.head is not None:
                encoder_outputs = self.head(
                    encoder_outputs, pixel_attention_mask
                )
            return encoder_outputs

        encoder_outputs = resolve_visual_encoder_outputs(
            encoder_outputs,
            None,
            select_layers=select_layers,
            max_possible_layers=self.config.num_hidden_layers,
            last_hs_proc=last_hs_proc,
            feature_select_strategy=feature_select_strategy,
        )

        return encoder_outputs

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        layer_count = len(self.encoder.layers)

        for name, loaded_weight in weights:
            if (
                name.startswith("post_layernorm")
                and self.post_layernorm is None
            ):
                continue

            if self.head is None and name.startswith("head"):
                continue

            if name.startswith("encoder.layers"):
                layer_idx = int(name.split(".")[2])
                if layer_idx >= layer_count:
                    continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(
                    param, "weight_loader", default_weight_loader
                )
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


# ==================== Text Transformer (reuse SigLIP v1 structure) ====================


class Siglip2TextTransformer(nn.Module):
    """SigLIP2 text transformer - same architecture as SigLIP v1."""

    def __init__(
        self,
        config: Siglip2TextConfig,
        quant_config: QuantizationConfig | None = None,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = Siglip2TextEmbeddings(config)

        self.encoder = Siglip2Encoder(
            config=config,
            quant_config=quant_config,
            prefix=f"{prefix}.encoder",
        )

        self.final_layer_norm = nn.LayerNorm(
            embed_dim, eps=config.layer_norm_eps
        )
        self.head = nn.Linear(embed_dim, config.projection_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings.token_embedding(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embeddings(input_ids, position_ids, inputs_embeds)

        last_hidden_state = self.encoder(
            inputs_embeds=hidden_states, return_all_hidden_states=False
        )

        last_hidden_state = self.final_layer_norm(last_hidden_state)

        return last_hidden_state

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(
                    param, "weight_loader", default_weight_loader
                )
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Siglip2TextEmbeddings(nn.Module):
    """SigLIP2 text embeddings - same structure as SigLIP v1."""

    def __init__(self, config: Siglip2TextConfig):
        super().__init__()
        self.config = config

        self.token_embedding = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )

        self.position_embedding = VocabParallelEmbedding(
            config.max_position_embeddings, config.hidden_size
        )

        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings


# ==================== Main Embedding Model ====================


@default_pooling_type(seq_pooling_type="CLS")
@MULTIMODAL_REGISTRY.register_processor(
    Siglip2MultiModalProcessor,
    info=Siglip2ProcessingInfo,
    dummy_inputs=Siglip2DummyInputsBuilder,
)
class Siglip2EmbeddingModel(nn.Module, SupportsMultiModal, SupportsQuant):
    is_pooling_model = True

    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return None
        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config: Siglip2Config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config

        if hasattr(config, "num_labels"):
            config.num_labels = 0

        text_config = config.text_config
        vision_config = config.vision_config

        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size
        self.text_projection_size = text_config.projection_size

        with self._mark_language_model(vllm_config):
            self.text_model = Siglip2TextTransformer(
                text_config,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "text_model"),
            )

        with self._mark_tower_model(vllm_config, "image"):
            self.vision_model = Siglip2VisionTransformer(
                vision_config,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "vision_model"),
                use_head=True,  # Attention pooling head reduces to 1 token
            )

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None
        self.pooler_config = pooler_config

        self.pooler = DispatchPooler.for_embedding(pooler_config)

        self._is_text_input = True

    def get_text_features(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        last_hidden_state = self.text_model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        text_features = self.text_model.head(last_hidden_state)

        # SigLIP2 uses reversed position_ids like SigLIP v1;
        # flip sequences to move EOS token to first position
        text_features = self._flip_sequences_by_position_ids(
            text_features, position_ids
        )

        return text_features

    def _flip_sequences_by_position_ids(
        self,
        features: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        if len(features) == 1:
            return features

        position_diffs = position_ids[1:] - position_ids[:-1]
        boundary_mask = position_diffs <= 0

        boundary_indices = torch.cat(
            [
                torch.tensor([0], device=features.device),
                torch.where(boundary_mask)[0] + 1,
                torch.tensor([len(features)], device=features.device),
            ]
        )

        lengths = boundary_indices[1:] - boundary_indices[:-1]
        starts = boundary_indices[:-1]
        ends = boundary_indices[1:]

        sequence_ids = torch.arange(
            len(lengths), device=features.device
        ).repeat_interleave(lengths)

        current_positions = torch.arange(
            len(features), device=features.device
        )
        flip_indices = (
            starts[sequence_ids]
            + ends[sequence_ids]
            - 1
            - current_positions
        )

        return features[flip_indices]

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        spatial_shapes: torch.LongTensor,
        pixel_attention_mask: torch.Tensor | None = None,
        feature_select_strategy: VisionFeatureSelectStrategy | None = None,
    ) -> torch.Tensor:
        if feature_select_strategy is None:
            feature_select_strategy = _get_vision_feature_select_strategy(
                self.pooler_config.seq_pooling_type
            )

        pooled_output = self.vision_model(
            pixel_values=pixel_values,
            spatial_shapes=spatial_shapes,
            pixel_attention_mask=pixel_attention_mask,
            select_layers=None,
            feature_select_strategy=feature_select_strategy,
        )

        return pooled_output

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> tuple[Siglip2ImagePixelInputs, torch.LongTensor, torch.Tensor | None] | None:
        pixel_values = kwargs.pop("pixel_values", None)
        if pixel_values is None:
            return None

        spatial_shapes = kwargs.pop("spatial_shapes", None)
        pixel_attention_mask = kwargs.pop("pixel_attention_mask", None)

        inputs = Siglip2ImagePixelInputs(
            type="pixel_values",
            data=pixel_values,
        )
        return inputs, spatial_shapes, pixel_attention_mask

    def _process_image_inputs(
        self,
        inputs: Siglip2ImagePixelInputs,
        spatial_shapes: torch.LongTensor,
        pixel_attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        pixel_values = inputs["data"]

        # Ensure batch dimension exists:
        # embed_multimodal may provide unbatched tensors
        if pixel_values.ndim == 2:
            # (max_patches, patch_dim) -> (1, max_patches, patch_dim)
            pixel_values = pixel_values.unsqueeze(0)
        if spatial_shapes.ndim == 1:
            # (2,) -> (1, 2)
            spatial_shapes = spatial_shapes.unsqueeze(0)
        if pixel_attention_mask is not None and pixel_attention_mask.ndim == 1:
            # (max_patches,) -> (1, max_patches)
            pixel_attention_mask = pixel_attention_mask.unsqueeze(0)

        return self.get_image_features(
            pixel_values,
            spatial_shapes,
            pixel_attention_mask,
        )

    def _embed_text_input_ids(
        self,
        input_ids: torch.Tensor,
        embed_input_ids: Callable[[torch.Tensor], torch.Tensor],
        *,
        is_multimodal: torch.Tensor | None,
        handle_oov_mm_token: bool,
    ) -> torch.Tensor:
        inputs_embeds = super()._embed_text_input_ids(
            input_ids,
            embed_input_ids,
            is_multimodal=is_multimodal,
            handle_oov_mm_token=handle_oov_mm_token,
        )

        inputs_embeds_size = self.text_projection_size
        if inputs_embeds.shape[1] < inputs_embeds_size:
            inputs_embeds = torch.cat(
                [
                    inputs_embeds,
                    inputs_embeds.new_empty(
                        inputs_embeds.shape[0],
                        inputs_embeds_size - inputs_embeds.shape[1],
                    ),
                ],
                dim=1,
            )
        elif inputs_embeds.shape[1] > inputs_embeds_size:
            raise NotImplementedError

        return inputs_embeds

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        self._is_text_input = (
            multimodal_embeddings is None
            or len(multimodal_embeddings) == 0
        )

        if multimodal_embeddings is None or is_multimodal is None:
            return super().embed_input_ids(input_ids)

        return super().embed_input_ids(
            input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
            handle_oov_mm_token=handle_oov_mm_token,
        )

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        result = self._parse_and_validate_image_input(**kwargs)
        if result is None:
            return []

        inputs, spatial_shapes, pixel_attention_mask = result
        vision_embeddings = self._process_image_inputs(
            inputs, spatial_shapes, pixel_attention_mask
        )
        return vision_embeddings

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            raise RuntimeError("PP is not supported for this model")

        # Multimodal inputs (image embeddings)
        if not self._is_text_input:
            return inputs_embeds

        hidden_size = self.text_embed_dim
        if inputs_embeds.shape[1] > hidden_size:
            inputs_embeds = inputs_embeds[:, :hidden_size]
        elif inputs_embeds.shape[1] < hidden_size:
            raise NotImplementedError

        return self.get_text_features(input_ids, positions, inputs_embeds)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(
            self,
            skip_substrs=[".position_ids"],
            ignore_unexpected_prefixes=["logit_scale.", "logit_bias."],
        )
        return loader.load_weights(weights)


# ==================== Image Classification Model ====================


@default_pooling_type(seq_pooling_type="CLS")
@MULTIMODAL_REGISTRY.register_processor(
    Siglip2MultiModalProcessor,
    info=Siglip2ProcessingInfo,
    dummy_inputs=Siglip2DummyInputsBuilder,
)
class Siglip2ClassificationModel(nn.Module, SupportsMultiModal, SupportsQuant):
    """SigLIP2 image classification model for vLLM.

    Serves ``Siglip2ForImageClassification`` via ``/v1/embeddings``.
    The returned "embedding" is a raw logit vector of shape ``(num_labels,)``;
    clients apply softmax to obtain class probabilities.

    Architecture (mirrors HF ``Siglip2ForImageClassification``):
        Image -> VisionEncoder (post_layernorm, NO attention pooling head)
              -> masked average pooling -> nn.Linear -> logits

    Because vLLM pre-allocates GPU buffers at ``hidden_size`` width, logits
    are zero-padded to ``hidden_size`` in :meth:`embed_input_ids` and
    truncated back to ``num_labels`` in :meth:`forward`.
    """

    is_pooling_model = True

    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return None
        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config: Siglip2Config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config

        vision_config = config.vision_config
        self.num_labels = config.num_labels
        self.hidden_size = vision_config.hidden_size
        self._pad_width = self.hidden_size - self.num_labels

        with self._mark_tower_model(vllm_config, "image"):
            self.vision_model = Siglip2VisionTransformer(
                vision_config,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "vision_model"),
                use_head=False,
            )

        self.classifier = nn.Linear(
            vision_config.hidden_size, self.num_labels
        )

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None
        self.pooler = DispatchPooler.for_embedding(pooler_config)

    # -- image parsing (reuses Siglip2EmbeddingModel pattern) --

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> tuple[Siglip2ImagePixelInputs, torch.LongTensor,
               torch.Tensor | None] | None:
        pixel_values = kwargs.pop("pixel_values", None)
        if pixel_values is None:
            return None

        spatial_shapes = kwargs.pop("spatial_shapes", None)
        pixel_attention_mask = kwargs.pop("pixel_attention_mask", None)

        return (
            Siglip2ImagePixelInputs(type="pixel_values", data=pixel_values),
            spatial_shapes,
            pixel_attention_mask,
        )

    def _get_image_features(
        self,
        pixel_values: torch.Tensor,
        spatial_shapes: torch.LongTensor,
        pixel_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Vision encoder -> masked average pooling -> classifier logits.

        Returns:
            ``(batch, 1, num_labels)`` — the extra dim-1 matches vLLM's
            expected per-item embedding shape.
        """
        # Ensure batch dimension (embed_multimodal may provide unbatched)
        if pixel_values.ndim == 2:
            pixel_values = pixel_values.unsqueeze(0)
        if spatial_shapes.ndim == 1:
            spatial_shapes = spatial_shapes.unsqueeze(0)
        if pixel_attention_mask is not None and pixel_attention_mask.ndim == 1:
            pixel_attention_mask = pixel_attention_mask.unsqueeze(0)

        # (batch, num_patches, hidden_size)
        hidden_states = self.vision_model(
            pixel_values=pixel_values,
            spatial_shapes=spatial_shapes,
            pixel_attention_mask=pixel_attention_mask,
        )

        # Masked average pooling (matching HF Siglip2ForImageClassification)
        if pixel_attention_mask is not None:
            mask = pixel_attention_mask[..., None].to(
                dtype=hidden_states.dtype, device=hidden_states.device
            )
            pooled = torch.sum(hidden_states * mask, dim=1) / mask.sum(dim=1)
        else:
            pooled = hidden_states.mean(dim=1)

        # (batch, hidden_size) -> (batch, 1, num_labels)
        return self.classifier(pooled).unsqueeze(1)

    # -- multimodal interface --

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        result = self._parse_and_validate_image_input(**kwargs)
        if result is None:
            return []

        inputs, spatial_shapes, pixel_attention_mask = result
        return self._get_image_features(
            inputs["data"], spatial_shapes, pixel_attention_mask
        )

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        """Merge multimodal logits into the token-embedding tensor.

        Image-only model: no language model exists. We concatenate the
        per-image logit tensors and zero-pad the last dimension from
        ``num_labels`` to ``hidden_size`` (vLLM pre-allocates buffers at
        ``hidden_size``).
        """
        if multimodal_embeddings is not None and len(multimodal_embeddings) > 0:
            if isinstance(multimodal_embeddings, torch.Tensor):
                emb = multimodal_embeddings
            else:
                emb = torch.cat(
                    [t if t.ndim == 2 else t.unsqueeze(0)
                     for t in multimodal_embeddings],
                    dim=0,
                )
            if self._pad_width > 0:
                emb = F.pad(emb, (0, self._pad_width))
            return emb

        return torch.zeros(
            len(input_ids), self.hidden_size,
            device=input_ids.device,
            dtype=self.classifier.weight.dtype,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            raise RuntimeError("PP is not supported for this model")

        # Strip the zero-padding added by embed_input_ids
        return inputs_embeds[..., :self.num_labels]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(
            self,
            skip_substrs=[".position_ids"],
        )
        return loader.load_weights(weights)
