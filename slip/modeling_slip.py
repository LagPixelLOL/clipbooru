import math
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import transformers
from transformers.modeling_outputs import ImageClassifierOutput
from typing import Optional, Union

def _vision_embedding_forward_patch(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
    batch_size = pixel_values.shape[0]
    if isinstance(self.patch_embedding, SLIPPatchEmbeddings):
        target_dtype = self.patch_embedding.conv_layers[0].weight.dtype
    else:
        target_dtype = self.patch_embedding.weight.dtype
    patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype)) # shape = [*, width, grid, grid]
    patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

    class_embeds = self.class_embedding.expand(batch_size, 1, -1)
    embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
    embeddings = embeddings + self.position_embedding(self.position_ids)
    return embeddings

transformers.models.clip.modeling_clip.CLIPVisionEmbeddings.forward = _vision_embedding_forward_patch

class SLIPPatchEmbeddings(nn.Module):

    def __init__(self, config: transformers.CLIPVisionConfig):
        super().__init__()

        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_side_len = self.image_size // self.patch_size

        # Calculate the number of Conv2d layers needed.
        num_layers = int(math.log2(self.image_size // self.patch_side_len))

        # Create a list of Conv2d layers.
        layers = []
        in_channels = 3 # Assuming RGB input.
        current_size = self.image_size
        for i in range(num_layers):
            out_channels = min(self.embed_dim, in_channels * 4)

            # Method A.
            # stride = 2 if current_size // 2 >= self.patch_side_len else 1
            # layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1))
            # layers.append(nn.ReLU())
            # in_channels = out_channels
            # if stride == 2:
            #     current_size //= 2

            # Method B.
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU())
            out_size = current_size // 2
            if out_size >= self.patch_side_len:
                layers.append(nn.MaxPool2d(2, 2))
                current_size = out_size
            in_channels = out_channels

        # Add a final Conv2d layer to adjust the number of channels into the embedding dimension.
        layers.append(nn.Conv2d(in_channels, self.embed_dim, kernel_size=1, stride=1))

        self.conv_layers = nn.Sequential(*layers)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        x = self.conv_layers(pixel_values)

        # Ensure the output has the correct spatial dimensions.
        if x.shape[-2:] != (self.patch_side_len, self.patch_side_len):
            x = nn.functional.adaptive_avg_pool2d(x, (self.patch_side_len, self.patch_side_len))

        return x

# Copied from CLIPForImageClassification into SLIPForImageClassification.
class SLIPForImageClassification(transformers.CLIPPreTrainedModel):
    main_input_name = "pixel_values"

    def __init__(self, config: transformers.CLIPConfig) -> None:
        super().__init__(config)

        self.config = config
        self.num_labels = config.num_labels
        vision_model = transformers.CLIPVisionModel._from_config(
            config.vision_config, attn_implementation=config._attn_implementation
        )
        self.vision_model = vision_model.vision_model

        # Monkey patching the patch embeddings.
        self.vision_model.embeddings.patch_embedding = SLIPPatchEmbeddings(config.vision_config)

        # Classifier head.
        self.classifier = (
            nn.Linear(config.vision_config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # Initialize weights and apply final processing.
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vision_model(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # average pool the patch tokens.
        sequence_output = torch.mean(sequence_output[:, 1:, :], dim=1)
        # apply classifier.
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism.
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor
        if isinstance(module, transformers.models.clip.modeling_clip.CLIPTextEmbeddings):
            module.token_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
            module.position_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
        elif isinstance(module, transformers.models.clip.modeling_clip.CLIPVisionEmbeddings):
            factor = self.config.initializer_factor
            nn.init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim**-0.5 * factor)
            if isinstance(module.patch_embedding, SLIPPatchEmbeddings):
                for conv_layer in module.patch_embedding.conv_layers:
                    if not isinstance(conv_layer, nn.Conv2d):
                        continue
                    nn.init.normal_(conv_layer.weight, std=module.config.initializer_range * factor)
            else:
                nn.init.normal_(module.patch_embedding.weight, std=module.config.initializer_range * factor)
            nn.init.normal_(module.position_embedding.weight, std=module.config.initializer_range * factor)
        elif isinstance(module, transformers.models.clip.modeling_clip.CLIPAttention):
            factor = self.config.initializer_factor
            in_proj_std = (module.embed_dim**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            out_proj_std = (module.embed_dim**-0.5) * factor
            nn.init.normal_(module.q_proj.weight, std=in_proj_std)
            nn.init.normal_(module.k_proj.weight, std=in_proj_std)
            nn.init.normal_(module.v_proj.weight, std=in_proj_std)
            nn.init.normal_(module.out_proj.weight, std=out_proj_std)
        elif isinstance(module, transformers.models.clip.modeling_clip.CLIPMLP):
            factor = self.config.initializer_factor
            in_proj_std = (module.config.hidden_size**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
            nn.init.normal_(module.fc1.weight, std=fc_std)
            nn.init.normal_(module.fc2.weight, std=in_proj_std)
        elif isinstance(module, transformers.models.clip.modeling_clip.CLIPModel):
            nn.init.normal_(
                module.text_projection.weight,
                std=module.text_embed_dim**-0.5 * self.config.initializer_factor,
            )
            nn.init.normal_(
                module.visual_projection.weight,
                std=module.vision_embed_dim**-0.5 * self.config.initializer_factor,
            )
        elif isinstance(module, transformers.models.clip.modeling_clip.CLIPVisionModelWithProjection):
            nn.init.normal_(
                module.visual_projection.weight,
                std=self.config.hidden_size**-0.5 * self.config.initializer_factor,
            )
        elif isinstance(module, transformers.models.clip.modeling_clip.CLIPTextModelWithProjection):
            nn.init.normal_(
                module.text_projection.weight,
                std=self.config.hidden_size**-0.5 * self.config.initializer_factor,
            )
        elif isinstance(module, transformers.models.clip.modeling_clip.CLIPForImageClassification):
            nn.init.normal_(
                module.classifier.weight,
                std=self.config.vision_config.hidden_size**-0.5 * self.config.initializer_factor,
            )

        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
