from typing import Optional

from aitemplate.compiler import ops
from aitemplate.frontend import nn, Tensor


class QuickGELUActivation(nn.Module):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, x):
        x1 = x * 1.702
        x1 = ops.sigmoid(x1)
        x = x * x1
        return x


class CLIPMLP(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer="GELU",
        drop=0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(
            in_features,
            hidden_features,
            specialization="gelu",
        )
        self.fc2 = nn.Linear(hidden_features, out_features, specialization="add")

    def forward(self, x, res):
        shape = x.shape()
        x = self.fc1(x)
        x = self.fc2(x, res)
        return ops.reshape()(x, shape)


class CLIPMLPQuickGelu(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(
            in_features,
            hidden_features,
        )
        self.activation_fn = QuickGELUActivation()

        self.fc2 = nn.Linear(hidden_features, out_features, specialization="add")

    def forward(self, x, res):
        # shape = get_shape(x)
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x, res)
        return ops.reshape()(x, x.shape())


class CLIPEncoderLayer(nn.Module):
    ACT_LAYER_TO_CLIP_MLP_MAP = {
        "gelu": CLIPMLP,
        "quick_gelu": CLIPMLPQuickGelu,
    }

    def __init__(
        self,
        hidden_size=768,
        num_attention_heads=12,
        attention_dropout=0.0,
        mlp_ratio=4.0,
        batch_size=1,
        seq_len=16,
        causal=False,
        mask_seq=0,
        act_layer="gelu",
    ):
        super().__init__()
        self.embed_dim = hidden_size
        self.self_attn = nn.CrossAttention(
            hidden_size,
            seq_len,
            seq_len,
            num_attention_heads,
            qkv_bias=True,
            causal=causal,
        )

        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.mlp = self.ACT_LAYER_TO_CLIP_MLP_MAP[act_layer](
            hidden_size, int(hidden_size * mlp_ratio)
        )
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: Tensor,
        output_attentions: Optional[bool] = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, hidden_states, hidden_states, residual
        )

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states, residual)

        return hidden_states


class CLIPEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`CLIPEncoderLayer`].
    Args:
        config: CLIPConfig
    """

    def __init__(
        self,
        num_hidden_layers=12,
        output_attentions=False,
        output_hidden_states=False,
        use_return_dict=False,
        hidden_size=768,
        num_attention_heads=12,
        batch_size=1,
        seq_len=64,
        causal=False,
        mask_seq=0,
        act_layer="gelu",
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                CLIPEncoderLayer(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    causal=causal,
                    mask_seq=mask_seq,
                    act_layer=act_layer,
                )
                for _ in range(num_hidden_layers)
            ]
        )
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.use_return_dict = use_return_dict

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[Tensor] = None,
        causal_attention_mask: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                [What are attention masks?](../glossary#attention-mask)
            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        encoder_states = () if output_hidden_states else None
        # all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for _, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            layer_outputs = encoder_layer(hidden_states)
            hidden_states = layer_outputs

        return hidden_states


class CLIPVisionEmbeddings(nn.Module):
    def __init__(
        self,
        hidden_size=768,
        num_channels=3,
        image_size=224,
        patch_size=16,
        dtype="float16",
    ):
        super().__init__()
        self.embed_dim = hidden_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels

        self.class_embedding = nn.Parameter(shape=[1, 1, hidden_size], dtype=dtype)
        num_channels = num_channels + (4 - (num_channels % 4))
        self.patch_embedding = nn.Conv2dBiasFewChannels(
            in_channels=num_channels,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            dtype=dtype,
        )

        self.num_patches = (image_size // patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(
            shape=[self.num_positions, hidden_size], dtype=dtype
        )

    def forward(self, pixel_values: Tensor, position_ids: Tensor) -> Tensor:
        pixel_values = ops.pad_last_dim(4, 4)(pixel_values)
        input_shape = ops.size()(pixel_values)
        batch_size = input_shape[0]
        patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds = ops.flatten(1, 2)(patch_embeds)

        class_embeds = ops.expand()(self.class_embedding.tensor(), [batch_size, 1, -1])
        class_embeds._attrs["shape"][0] = pixel_values._attrs["shape"][0]
        embeddings = ops.concatenate()([class_embeds, patch_embeds], dim=1)

        position_embedding = self.position_embedding.tensor()
        position_embedding = ops.reshape()(
            position_embedding, [1, self.num_positions, self.embed_dim]
        )
        position_embedding = ops.expand()(position_embedding, [input_shape[0], -1, -1])

        embeddings = embeddings + ops.batch_gather()(position_embedding, position_ids)

        return embeddings


class CLIPVisionTransformer(nn.Module):
    def __init__(
        self,
        hidden_size=1024,
        layer_norm_eps=1e-05,
        num_channels=3,
        image_size=224,
        patch_size=14,
        num_hidden_layers=24,
        num_attention_heads=16,
        hidden_act="quick_gelu",
        projection_dim=None,
    ):
        super().__init__()
        self.embed_dim = hidden_size
        self.embeddings = CLIPVisionEmbeddings(
            hidden_size=hidden_size,
            num_channels=num_channels,
            image_size=image_size,
            patch_size=patch_size,
        )
        self.pre_layrnorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.encoder = CLIPEncoder(
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            act_layer=hidden_act,
        )
        self.post_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        if projection_dim is not None:
            self.visual_projection = nn.Linear(hidden_size, projection_dim, bias=False)
        else:
            self.visual_projection = None

    def forward(
        self,
        pixel_values: Tensor,
        position_ids: Tensor,
    ):
        batch = ops.size()(pixel_values)[0]._attrs["int_var"]._attrs["values"][0]
        hidden_states = self.embeddings(pixel_values, position_ids)
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
        )

        last_hidden_state = encoder_outputs
        pooled_output = ops.dynamic_slice()(
            last_hidden_state,
            start_indices=[0, 0, 0],
            end_indices=[batch, 1, self.embed_dim],
        )
        pooled_output = self.post_layernorm(pooled_output)
        pooled_output = ops.squeeze(dim=0)(pooled_output)
        if self.visual_projection is not None:
            image_embeds = self.visual_projection(pooled_output)
            image_embeds._attrs["is_output"] = True
            image_embeds._attrs["name"] = "image_embeds"
            return image_embeds
        else:
            pooled_output._attrs["is_output"] = True
            pooled_output._attrs["name"] = "pooled_output"
            last_hidden_state._attrs["is_output"] = True
            last_hidden_state._attrs["name"] = "last_hidden_state"
            return pooled_output, last_hidden_state