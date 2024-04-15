# coding=utf-8
# Copyright 2024 the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch Llava-NeXT model."""

from typing import List, Optional

import torch
from torch import nn
# TODO(xwjiang): We should port CLIPVisionModel's code over to not depend on
# transformers' impl.
from transformers import CLIPVisionModel, LlavaNextConfig

from vllm.attention import AttentionMetadata
from vllm.config import VisionLanguageConfig
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)
from vllm.sequence import SamplerOutput

_KEYS_TO_MODIFY_MAPPING = {
    "language_model.lm_head": "lm_head",
    "language_model.model": "language_model",
}

LLAVA_NEXT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "llava-hf/llava-v1.6-mistral-7b-hf",
    # See all LLaVA-NeXT models at https://huggingface.co/models?filter=llava_next
]

# Copied from https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/image_processing_utils.py
def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (`tuple`):
            The size of the input image in the format (width, height).
        grid_pinpoints (`List`):
            A list containing possible resolutions. Each item in the list should be a tuple or list
            of the form `(height, width)`.
        patch_size (`int`):
            The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if not isinstance(grid_pinpoints, list):
        raise ValueError("grid_pinpoints should be a list of tuples or lists")

    height, width = select_best_resolution(image_size, grid_pinpoints)
    return height // patch_size, width // patch_size

# Copied from https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/image_processing_utils.py
def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
        tensor (`torch.Tensor`):
            The image tensor, assumed to be of shape (num_channels, height, width).
        original_size (`tuple`):
            The original size of the image (height, width).

    Returns:
        `torch.Tensor`: The unpadded image tensor.
    """
    original_height, original_width = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor

# Copied from https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/image_processing_utils.py
def select_best_resolution(original_size: tuple, possible_resolutions: list) -> tuple:
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    This is done by calculating the effective and wasted resolution for each possible resolution.

    The best fit resolution is the one that maximizes the effective resolution and minimizes the wasted resolution.

    Args:
        original_size (tuple):
            The original size of the image in the format (height, width).
        possible_resolutions (list):
            A list of possible resolutions in the format [(height1, width1), (height2, width2), ...].

    Returns:
        tuple: The best fit resolution in the format (height, width).
    """
    original_height, original_width = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for height, width in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (height, width)

    return best_fit


# Copied from vllm/model_executor/models/llava.py with Llava->LlavaNext
class LlavaNextMultiModalProjector(nn.Module):
    def __init__(self, vision_hidden_size: int, text_hidden_size: int,
                 projector_hidden_act: str):
        super().__init__()

        self.linear_1 = nn.Linear(vision_hidden_size,
                                  text_hidden_size,
                                  bias=True)
        self.act = get_act_fn(projector_hidden_act)
        self.linear_2 = nn.Linear(text_hidden_size,
                                  text_hidden_size,
                                  bias=True)

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states

# Copied from vllm/model_executor/models/llava.py
def _merge_vision_embeddings(input_ids: torch.Tensor,
                             inputs_embeds: torch.Tensor,
                             vision_embeddings: torch.Tensor,
                             image_token_id: int):
    """In place merges in vision_embeddings with inputs_embeds."""


    mask = (input_ids == image_token_id)
    inputs_embeds[mask] = vision_embeddings.view(-1,
                                                 vision_embeddings.shape[-1])


class LlavaNextForConditionalGeneration(nn.Module):

    def __init__(self,
                 config: LlavaNextConfig,
                 vision_language_config: VisionLanguageConfig,
                 linear_method: Optional["LinearMethodBase"] = None) -> None:
        super().__init__()
        self.config = config
        self.image_newline = nn.Parameter(torch.empty(config.text_config.hidden_size, dtype=self.config.text_config.torch_dtype))
        self.vision_language_config = vision_language_config

        assert self.vision_language_config, (
            "Provide `image_input_type` and other vision "
            "related configurations through LLM entrypoint "
            "or engine arguments.")

        if self.vision_language_config.image_input_type == (
                VisionLanguageConfig.ImageInputType.PIXEL_VALUES):
            self.vision_tower = CLIPVisionModel(config.vision_config)
        else:
            self.vision_tower = None

        self.multi_modal_projector = LlavaNextMultiModalProjector(
            vision_hidden_size=config.vision_config.hidden_size,
            text_hidden_size=config.text_config.hidden_size,
            projector_hidden_act=config.projector_hidden_act)

        self.linear_method = linear_method
        self.language_model = LlamaModel(config.text_config, linear_method)
        self.unpadded_vocab_size = config.text_config.vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.text_config.hidden_size,
            org_num_embeddings=self.language_model.org_vocab_size)
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size, logit_scale)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        image_input: Optional[torch.Tensor] = None,
        image_size: Optional[torch.Tensor] = None,
    ) -> SamplerOutput:

        # 1. Extract the input embeddings
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if image_input is not None:
            if self.vision_tower is not None:
                batch_size, num_patches, num_channels, height, width = image_input.shape
                reshaped_pixel_values = image_input.view(batch_size * num_patches, num_channels, height, width)
                image_outputs = self.vision_tower(reshaped_pixel_values, output_hidden_states=True)
                image_features = image_outputs.hidden_states[self.config.vision_feature_layer]

                if self.config.vision_feature_select_strategy == "default":
                    image_features = image_features[:, 1:]
                elif self.config.vision_feature_select_strategy == "full":
                    image_features = image_features
                else:
                    raise ValueError(
                        f"Unexpected select feature strategy: "
                        f"{self.config.vision_feature_select_strategy}")

                image_features = self.multi_modal_projector(image_features)

                # split up image_features for each of the individual images
                # hence we get a list of image_features, each of shape (5, num_patches, hidden_size)
                # if we assume each image has 5 image features (base image + 4 patches)
                split_sizes = [image.shape[0] for image in image_input]
                image_features = torch.split(image_features, split_sizes, dim=0)

                # NOTE we only support multimodal_patch_merge_type == "spatial_unpad"
                height = width = self.config.vision_config.image_size // self.config.vision_config.patch_size

                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]

                        if height * width != base_image_feature.shape[0]:
                            raise ValueError("The number of patches is not consistent with the image size.")
                        num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                            image_size[image_idx],
                            self.config.image_grid_pinpoints,
                            self.config.vision_config.image_size,
                        )
                        image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                        image_feature = unpad_image(image_feature, image_size[image_idx])
                        image_feature = torch.cat(
                            (
                                image_feature,
                                self.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1),
                            ),
                            dim=-1,
                        )
                        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        image_feature = torch.cat((image_feature, self.image_newline[None]), dim=0)
                    new_image_features.append(image_feature)
                vision_embeddings = torch.stack(new_image_features, dim=0)
                _merge_vision_embeddings(
                    input_ids, inputs_embeds, vision_embeddings,
                    self.vision_language_config.image_token_id)
                input_ids = None
            else:
                inputs_embeds = None

            hidden_states = self.language_model(input_ids,
                                                positions,
                                                kv_caches,
                                                attn_metadata,
                                                inputs_embeds=inputs_embeds)

        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                        sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head.weight, hidden_states,
                                        sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self,
                        model_name_or_path: str,
                        cache_dir: Optional[str] = None,
                        load_format: str = "auto",
                        revision: Optional[str] = None):
        # only doing this for language model part for now.
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):
            if "rotary_emb.inv_freq" in name:
                continue
            if "image_newline" in name:
                continue
            for key_to_modify, new_key in _KEYS_TO_MODIFY_MAPPING.items():
                if key_to_modify in name:
                    name = name.replace(key_to_modify, new_key)
            use_default_weight_loading = False
            if "vision" in name:
                if self.vision_tower is not None:
                    # We only do sharding for language model and
                    # not vision model for now.
                    use_default_weight_loading = True
            else:
                for (param_name, weight_name,
                        shard_id) in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    param = params_dict[name.replace(weight_name, param_name)]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    break
                else:
                    use_default_weight_loading = True
            if use_default_weight_loading:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
