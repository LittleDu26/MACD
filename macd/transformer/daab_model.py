import math
import torch
import torch.nn as nn


VOXEL_EMPTY = 0
VOXEL_RIGID = 1
VOXEL_SOFT = 2
VOXEL_H_ACT = 3
VOXEL_V_ACT = 4


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def silu(x):
    return x * torch.sigmoid(x)


def check_finite_tensor(name, tensor, controller_name="daab"):
    if tensor is not None and not torch.isfinite(tensor).all():
        raise FloatingPointError(
            "{} controller produced non-finite values at {}".format(controller_name, name)
        )


def split_daab_observation(obs):
    x_obs = obs[..., [0, 2, 4, 6, 8, 10]]
    y_obs = obs[..., [1, 3, 5, 7, 9, 11]]
    return x_obs, y_obs


def scale_daab_observation(obs, center_scale=1.0, velocity_scale=1.0):
    if center_scale == 1.0 and velocity_scale == 1.0:
        return obs
    scaled = obs.clone()
    scaled[..., 8:10] = scaled[..., 8:10] * center_scale
    scaled[..., 10:12] = scaled[..., 10:12] * velocity_scale
    return scaled


def make_silu_mlp(dims, final_activation=False):
    layers = []
    for idx, (dim_in, dim_out) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(dim_in, dim_out))
        if idx < len(dims) - 2 or final_activation:
            layers.append(SiLU())
    return nn.Sequential(*layers)


def infer_axis_embedding_dim(embedding_dim):
    if embedding_dim % 2 != 0:
        raise ValueError("DAAB axis concat/factorized embedding requires an even embedding_dim")
    return embedding_dim // 2


class TypeSpecificAxisEncoder(nn.Module):
    def __init__(self, axis_embedding_dim=None, embedding_dim=64):
        super(TypeSpecificAxisEncoder, self).__init__()
        if axis_embedding_dim is None:
            axis_embedding_dim = infer_axis_embedding_dim(embedding_dim)
        self.axis_embedding_dim = axis_embedding_dim
        self.embedding_dim = embedding_dim
        self.x_encoders = nn.ModuleList(
            [make_silu_mlp([6, axis_embedding_dim, axis_embedding_dim]) for _ in range(5)]
        )
        self.y_encoders = nn.ModuleList(
            [make_silu_mlp([6, axis_embedding_dim, axis_embedding_dim]) for _ in range(5)]
        )

    def forward(self, obs, voxel_type, obs_mask):
        x_obs, y_obs = split_daab_observation(obs)
        out_x = obs.new_zeros(obs.shape[:-1] + (self.axis_embedding_dim,))
        out_y = obs.new_zeros(obs.shape[:-1] + (self.axis_embedding_dim,))
        voxel_type = voxel_type.long()

        for voxel_id in (VOXEL_RIGID, VOXEL_SOFT, VOXEL_H_ACT, VOXEL_V_ACT):
            mask = (voxel_type == voxel_id) & (~obs_mask.bool())
            if mask.any():
                out_x[mask] = self.x_encoders[voxel_id](x_obs[mask])
                out_y[mask] = self.y_encoders[voxel_id](y_obs[mask])

        return torch.cat([out_x, out_y], dim=-1)


class TypeSpecificFullObsEncoder(nn.Module):
    def __init__(self, embedding_dim=64):
        super(TypeSpecificFullObsEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.encoders = nn.ModuleList(
            [make_silu_mlp([12, embedding_dim, embedding_dim]) for _ in range(5)]
        )

    def forward(self, obs, voxel_type, obs_mask):
        encoded = obs.new_zeros(obs.shape[:-1] + (self.embedding_dim,))
        voxel_type = voxel_type.long()

        for voxel_id in (VOXEL_RIGID, VOXEL_SOFT, VOXEL_H_ACT, VOXEL_V_ACT):
            mask = (voxel_type == voxel_id) & (~obs_mask.bool())
            if mask.any():
                encoded[mask] = self.encoders[voxel_id](obs[mask])

        return encoded


def make_daab_observation_encoder(args):
    mode = getattr(args, "daab_observation_encoder_mode", "axis_factorized")
    if mode == "axis_factorized":
        return TypeSpecificAxisEncoder(
            infer_axis_embedding_dim(args.attention_embedding_size),
            args.attention_embedding_size,
        )
    if mode == "full_obs":
        return TypeSpecificFullObsEncoder(args.attention_embedding_size)
    raise ValueError("Unknown DAAB observation encoder mode: {}".format(mode))


class TypeSpecificDesignEncoder(nn.Module):
    def __init__(self, input_dim=9, embedding_dim=64):
        super(TypeSpecificDesignEncoder, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.encoders = nn.ModuleList(
            [make_silu_mlp([input_dim, embedding_dim, embedding_dim]) for _ in range(5)]
        )

    def forward(self, design_obs, voxel_type):
        design_obs = design_obs / 3.0
        encoded = design_obs.new_zeros(design_obs.shape[:-1] + (self.embedding_dim,))
        voxel_type = voxel_type.long().clamp(min=VOXEL_EMPTY, max=VOXEL_V_ACT)

        for voxel_id in (VOXEL_EMPTY, VOXEL_RIGID, VOXEL_SOFT, VOXEL_H_ACT, VOXEL_V_ACT):
            mask = voxel_type == voxel_id
            if mask.any():
                encoded[mask] = self.encoders[voxel_id](design_obs[mask])

        return encoded


class AxisPositionEmbedding(nn.Module):
    def __init__(
        self,
        max_coord,
        embedding_dim=64,
        mode="axis_concat",
    ):
        super(AxisPositionEmbedding, self).__init__()
        if mode not in ("axis_concat", "axis_sum"):
            raise ValueError("Unknown DAAB position embedding mode: {}".format(mode))
        self.mode = mode
        self.embedding_dim = embedding_dim
        self.axis_embedding_dim = infer_axis_embedding_dim(embedding_dim)
        pos_dim = self.axis_embedding_dim if mode == "axis_concat" else embedding_dim
        self.x_embedding = nn.Embedding(max_coord, pos_dim)
        self.y_embedding = nn.Embedding(max_coord, pos_dim)

    def forward(self, grid_pos):
        grid_pos = grid_pos.long()
        x_pos = grid_pos[..., 0].clamp(min=0, max=self.x_embedding.num_embeddings - 1)
        y_pos = grid_pos[..., 1].clamp(min=0, max=self.y_embedding.num_embeddings - 1)
        x_emb = self.x_embedding(x_pos)
        y_emb = self.y_embedding(y_pos)
        if self.mode == "axis_concat":
            return torch.cat([x_emb, y_emb], dim=-1)
        return x_emb + y_emb


class DAABFeatureBuilder(nn.Module):
    def __init__(self, feature_type="10d"):
        super(DAABFeatureBuilder, self).__init__()
        if feature_type not in (
            "10d",
            "10d_abs",
            "10d_direct",
            "5d",
            "5d_abs",
            "5d_direct",
        ):
            raise ValueError("Unknown DAAB bias feature type: {}".format(feature_type))
        self.feature_type = feature_type
        self.feature_dim = 5 if feature_type in ("5d", "5d_abs", "5d_direct") else 10

    def forward(self, obs):
        rel_corners = obs[..., 0:8].reshape(obs.shape[0], obs.shape[1], 4, 2)
        centers = obs[..., 8:10]
        corners = rel_corners + centers.unsqueeze(2)

        center_diff = centers.unsqueeze(2) - centers.unsqueeze(1)
        corner_diff = corners.unsqueeze(2) - corners.unsqueeze(1)
        if self.feature_type in ("10d", "10d_abs", "10d_direct"):
            if self.feature_type == "10d_direct":
                return torch.cat(
                    [
                        center_diff,
                        corner_diff.reshape(obs.shape[0], obs.shape[1], obs.shape[1], 8),
                    ],
                    dim=-1,
                )
            if self.feature_type == "10d_abs":
                center_diff = torch.abs(center_diff)
            corner_diff = torch.abs(corner_diff).reshape(obs.shape[0], obs.shape[1], obs.shape[1], 8)
            return torch.cat([center_diff, corner_diff], dim=-1)

        center_dist = torch.linalg.norm(center_diff, dim=-1, keepdim=True)
        corner_dist = torch.linalg.norm(corner_diff, dim=-1)
        if self.feature_type in ("5d", "5d_direct"):
            center_dist = torch.sign(center_diff[..., 0:1]) * center_dist
        if self.feature_type == "5d_direct":
            corner_dist = torch.sign(corner_diff[..., 0]) * corner_dist
        return torch.cat([center_dist, corner_dist], dim=-1)


class DAABAttention(nn.Module):
    def __init__(self, embedding_dim=64, heads=4, dropout=0.0):
        super(DAABAttention, self).__init__()
        if embedding_dim % heads != 0:
            raise ValueError("embedding_dim must be divisible by heads")
        self.embedding_dim = embedding_dim
        self.heads = heads
        self.head_dim = embedding_dim // heads
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, daab_bias, key_padding_mask=None, return_attention=False):
        seq_len, batch_size, _ = src.shape
        src_b = src.permute(1, 0, 2)
        q = self.q_proj(src_b).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(src_b).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(src_b).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        logits = logits + daab_bias
        if key_padding_mask is not None:
            key_mask = key_padding_mask.bool().unsqueeze(1).unsqueeze(2)
            logits = logits.masked_fill(key_mask, -1e9)

        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embedding_dim)
        output = self.out_proj(output).permute(1, 0, 2)

        if return_attention:
            return output, attn
        return output, None


class DAABEncoderLayer(nn.Module):
    def __init__(self, embedding_dim=64, heads=4, hidden_size=128, dropout=0.0):
        super(DAABEncoderLayer, self).__init__()
        self.self_attn = DAABAttention(embedding_dim, heads, dropout)
        self.linear1 = nn.Linear(embedding_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, embedding_dim)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, daab_bias, key_padding_mask=None, return_attention=False):
        src2 = self.norm1(src)
        attn_out, attn = self.self_attn(
            src2, daab_bias, key_padding_mask=key_padding_mask, return_attention=return_attention
        )
        src = src + self.dropout1(attn_out)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(silu(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src, attn


class DAABTransformerEncoder(nn.Module):
    def __init__(self, embedding_dim=64, heads=4, hidden_size=128, layers=1, dropout=0.0):
        super(DAABTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                DAABEncoderLayer(embedding_dim, heads, hidden_size, dropout)
                for _ in range(layers)
            ]
        )

    def forward(self, src, daab_bias, key_padding_mask=None, return_attention=False,
                return_hidden_states=False):
        attention_maps = []
        hidden_states = []
        output = src
        for layer in self.layers:
            output, attn = layer(
                output, daab_bias, key_padding_mask=key_padding_mask, return_attention=return_attention
            )
            if return_attention:
                attention_maps.append(attn)
            if return_hidden_states:
                hidden_states.append(output)
        if return_hidden_states:
            return output, attention_maps if return_attention else None, hidden_states
        return output, attention_maps if return_attention else None


class DAABBias(nn.Module):
    def __init__(
        self,
        heads=4,
        bias_scale=1.0,
        bias_clip=10.0,
        bias_layers=2,
        hidden_dim=64,
        feature_type="10d",
    ):
        super(DAABBias, self).__init__()
        if bias_layers < 1 or bias_layers > 3:
            raise ValueError("DAAB bias layers must be between 1 and 3, got {}".format(bias_layers))
        if hidden_dim <= 0:
            raise ValueError("DAAB bias hidden dim must be positive, got {}".format(hidden_dim))
        self.feature_builder = DAABFeatureBuilder(feature_type)
        self.bias_scale = bias_scale
        self.bias_clip = bias_clip
        self.bias_layers = bias_layers
        self.hidden_dim = hidden_dim
        self.feature_type = feature_type
        dims = [self.feature_builder.feature_dim] + [hidden_dim] * (bias_layers - 1) + [heads]
        self.daab_mlp = make_silu_mlp(dims)
        nn.init.zeros_(self.daab_mlp[-1].weight)
        nn.init.zeros_(self.daab_mlp[-1].bias)

    def forward(self, obs):
        features = self.feature_builder(obs)
        bias = self.daab_mlp(features)
        if self.bias_clip is not None and self.bias_clip > 0:
            bias = bias.clamp(-self.bias_clip, self.bias_clip)
        bias = bias * self.bias_scale
        return bias.permute(0, 3, 1, 2)


class DAABActor(nn.Module):
    def __init__(self, sequence_size, other_feature_size, args):
        super(DAABActor, self).__init__()
        self.sequence_size = sequence_size
        self.embedding_dim = args.attention_embedding_size
        self.observation_encoder_mode = getattr(args, "daab_observation_encoder_mode", "axis_factorized")
        self.position_embedding_mode = getattr(args, "daab_position_embedding_mode", "axis_concat")
        self.center_scale = getattr(args, "daab_center_scale", 1.0)
        self.velocity_scale = getattr(args, "daab_velocity_scale", 1.0)
        self.encoder = make_daab_observation_encoder(args)
        self.pos_embedding = AxisPositionEmbedding(
            sequence_size,
            self.embedding_dim,
            self.position_embedding_mode,
        )
        heads = getattr(args, "daab_attention_heads", args.attention_heads)
        self.daab_bias = DAABBias(
            heads=heads,
            bias_scale=getattr(args, "daab_bias_scale", 1.0),
            bias_clip=getattr(args, "daab_bias_clip", 10.0),
            bias_layers=getattr(args, "daab_bias_layers", 2),
            hidden_dim=getattr(args, "daab_bias_hidden_dim", 64),
            feature_type=getattr(args, "daab_bias_feature_type", "10d"),
        )
        self.transformer = DAABTransformerEncoder(
            embedding_dim=self.embedding_dim,
            heads=heads,
            hidden_size=args.attention_hidden_size,
            layers=args.attention_layers,
            dropout=args.dropout_rate,
        )
        decoder_input_dim = self.embedding_dim + other_feature_size
        if args.condition_decoder:
            decoder_input_dim += 12
        self.decoder_h = nn.Sequential(
            nn.Linear(decoder_input_dim, 64), SiLU(), nn.Linear(64, 1), nn.Tanh()
        )
        self.decoder_v = nn.Sequential(
            nn.Linear(decoder_input_dim, 64), SiLU(), nn.Linear(64, 1), nn.Tanh()
        )
        self.condition_decoder = args.condition_decoder

    def reset_seq_size(self, sequence_size):
        self.sequence_size = sequence_size
        self.pos_embedding = AxisPositionEmbedding(
            sequence_size,
            self.embedding_dim,
            self.position_embedding_mode,
        )

    def forward(self, obs, other_state, voxel_type, grid_pos, token_index, obs_mask,
                return_attention=False, return_hidden_states=False):
        obs = scale_daab_observation(obs, self.center_scale, self.velocity_scale)
        encoded = self.encoder(obs, voxel_type, obs_mask) * math.sqrt(self.embedding_dim)
        encoded = encoded + self.pos_embedding(grid_pos)
        daab_bias = self.daab_bias(obs)
        transformer_result = self.transformer(
            encoded.permute(1, 0, 2), daab_bias, key_padding_mask=obs_mask,
            return_attention=return_attention, return_hidden_states=return_hidden_states,
        )
        if return_hidden_states:
            z, attention_maps, hidden_states = transformer_result
        else:
            z, attention_maps = transformer_result
        z = z.permute(1, 0, 2)
        other = other_state.unsqueeze(1).repeat(1, self.sequence_size, 1)
        decoder_input = torch.cat([z, other, obs], dim=-1) if self.condition_decoder else torch.cat([z, other], dim=-1)

        token_output = obs.new_zeros(obs.shape[0], self.sequence_size)
        h_mask = (voxel_type.long() == VOXEL_H_ACT) & (~obs_mask.bool())
        v_mask = (voxel_type.long() == VOXEL_V_ACT) & (~obs_mask.bool())
        if h_mask.any():
            token_output[h_mask] = self.decoder_h(decoder_input[h_mask]).squeeze(-1)
        if v_mask.any():
            token_output[v_mask] = self.decoder_v(decoder_input[v_mask]).squeeze(-1)

        output = obs.new_zeros(obs.shape[0], self.sequence_size)
        scatter_index = token_index.long().clamp(min=0, max=self.sequence_size - 1)
        output.scatter_(1, scatter_index, token_output)
        check_finite_tensor("actor_mean", output)
        if return_hidden_states:
            return output, attention_maps, hidden_states
        return output, attention_maps


class DAABCritic(nn.Module):
    def __init__(self, sequence_size, other_feature_size, args):
        super(DAABCritic, self).__init__()
        self.sequence_size = sequence_size
        self.embedding_dim = args.attention_embedding_size
        self.observation_encoder_mode = getattr(args, "daab_observation_encoder_mode", "axis_factorized")
        self.position_embedding_mode = getattr(args, "daab_position_embedding_mode", "axis_concat")
        self.design_input_dim = getattr(args, "number_neighbors", 9)
        self.center_scale = getattr(args, "daab_center_scale", 1.0)
        self.velocity_scale = getattr(args, "daab_velocity_scale", 1.0)
        self.encoder = make_daab_observation_encoder(args)
        self.pos_embedding = AxisPositionEmbedding(
            sequence_size,
            self.embedding_dim,
            self.position_embedding_mode,
        )
        heads = getattr(args, "daab_attention_heads", args.attention_heads)
        self.heads = heads
        self.daab_bias = DAABBias(
            heads=heads,
            bias_scale=getattr(args, "daab_bias_scale", 1.0),
            bias_clip=getattr(args, "daab_bias_clip", 10.0),
            bias_layers=getattr(args, "daab_bias_layers", 2),
            hidden_dim=getattr(args, "daab_bias_hidden_dim", 64),
            feature_type=getattr(args, "daab_bias_feature_type", "10d"),
        )
        self.transformer = DAABTransformerEncoder(
            embedding_dim=self.embedding_dim,
            heads=heads,
            hidden_size=args.attention_hidden_size,
            layers=args.attention_layers,
            dropout=args.dropout_rate,
        )
        decoder_input_dim = self.embedding_dim + other_feature_size
        if args.condition_decoder:
            decoder_input_dim += 12
        self.value_decoders = nn.ModuleList(
            [make_silu_mlp([decoder_input_dim, 64, 1]) for _ in range(5)]
        )
        self.condition_decoder = args.condition_decoder

    def reset_seq_size(self, sequence_size):
        self.sequence_size = sequence_size
        self.pos_embedding = AxisPositionEmbedding(
            sequence_size,
            self.embedding_dim,
            self.position_embedding_mode,
        )

    def _design_voxel_type(self, design_obs):
        center_index = 4
        if design_obs.shape[-1] <= center_index:
            raise ValueError(
                "DAAB design critic expects at least 5 design-neighbor features, got {}".format(
                    design_obs.shape[-1]
                )
            )
        return design_obs[..., center_index]

    def _build_design_daab_obs(self, design_obs, daab_obs):
        if self.design_input_dim != 9:
            raise ValueError(
                "DAAB design critic requires number_neighbors == 9, got {}".format(
                    self.design_input_dim
                )
            )
        if design_obs.dim() == 2:
            design_obs = design_obs.unsqueeze(0)
        if daab_obs.dim() == 1:
            daab_obs = daab_obs.unsqueeze(0)
        batch_size = design_obs.shape[0]
        if daab_obs.dim() == 2 and daab_obs.shape[0] != batch_size:
            daab_obs = daab_obs.unsqueeze(0)
        design_obs = design_obs.reshape(batch_size, self.sequence_size, -1)
        daab_obs = daab_obs.reshape(batch_size, self.sequence_size, -1)
        if design_obs.shape[-1] != 9:
            raise ValueError(
                "DAAB design critic requires 9-neighbor design inputs, got {}".format(
                    design_obs.shape[-1]
                )
            )
        if daab_obs.shape[-1] != 12:
            raise ValueError("Expected 12D DAAB obs, got {}".format(daab_obs.shape[-1]))

        design_daab_obs = daab_obs.clone()
        neighbor_indices = torch.tensor(
            [0, 1, 2, 3, 5, 6, 7, 8], device=design_obs.device
        )
        design_daab_obs[..., :8] = design_obs.index_select(-1, neighbor_indices)
        return design_daab_obs

    def forward_design(
        self,
        design_obs,
        daab_obs,
        other_state,
        voxel_type,
        grid_pos,
        obs_mask,
        return_attention=False,
        return_hidden_states=False,
    ):
        if design_obs.dim() == 2:
            design_obs = design_obs.unsqueeze(0)
        batch_size = design_obs.shape[0]
        design_obs = design_obs.reshape(batch_size, self.sequence_size, -1)
        if design_obs.shape[-1] != 9:
            raise ValueError(
                "DAAB design critic requires 9-neighbor design inputs, got {}".format(
                    design_obs.shape[-1]
                )
            )

        design_daab_obs = self._build_design_daab_obs(design_obs, daab_obs)
        design_voxel_type = self._design_voxel_type(design_obs).long().clamp(
            min=VOXEL_EMPTY, max=VOXEL_V_ACT
        )
        design_mask = design_voxel_type == VOXEL_EMPTY
        grid_pos = grid_pos.reshape(batch_size, self.sequence_size, 2)
        other_state = other_state.reshape(batch_size, -1)

        result = self.forward(
            design_daab_obs,
            other_state,
            design_voxel_type,
            grid_pos,
            design_mask,
            return_attention=return_attention,
            return_hidden_states=return_hidden_states,
            daab_bias_obs=daab_obs,
        )
        if return_hidden_states:
            value, attention_maps, hidden_states = result
        else:
            value, attention_maps = result
        check_finite_tensor("design_critic_value", value)
        if return_hidden_states:
            return value, attention_maps, hidden_states
        return value, attention_maps

    def forward(
        self,
        obs,
        other_state,
        voxel_type,
        grid_pos,
        obs_mask,
        return_attention=False,
        return_hidden_states=False,
        daab_bias_obs=None,
    ):
        # The design critic replaces obs[..., :8] with material-neighborhood
        # features for its encoder.  DAAB attention must still be constructed
        # from the physical voxel geometry, so it receives the untouched
        # 12-D DAAB observation through daab_bias_obs.
        if daab_bias_obs is None:
            daab_bias_obs = obs
        if daab_bias_obs.dim() == 2:
            daab_bias_obs = daab_bias_obs.reshape(obs.shape[0], self.sequence_size, -1)
        if daab_bias_obs.shape != obs.shape:
            raise ValueError(
                "DAAB bias observation shape {} does not match critic observation shape {}".format(
                    tuple(daab_bias_obs.shape), tuple(obs.shape)
                )
            )

        obs = scale_daab_observation(obs, self.center_scale, self.velocity_scale)
        daab_bias_obs = scale_daab_observation(
            daab_bias_obs, self.center_scale, self.velocity_scale
        )
        encoded = self.encoder(obs, voxel_type, obs_mask) * math.sqrt(self.embedding_dim)
        encoded = encoded + self.pos_embedding(grid_pos)
        daab_bias = self.daab_bias(daab_bias_obs)
        transformer_result = self.transformer(
            encoded.permute(1, 0, 2), daab_bias, key_padding_mask=obs_mask,
            return_attention=return_attention, return_hidden_states=return_hidden_states,
        )
        if return_hidden_states:
            z, attention_maps, hidden_states = transformer_result
        else:
            z, attention_maps = transformer_result
        z = z.permute(1, 0, 2)
        other = other_state.unsqueeze(1).repeat(1, self.sequence_size, 1)
        decoder_input = torch.cat([z, other, obs], dim=-1) if self.condition_decoder else torch.cat([z, other], dim=-1)

        token_values = obs.new_zeros(obs.shape[0], self.sequence_size)
        voxel_type_long = voxel_type.long()
        valid = ~obs_mask.bool()
        for voxel_id in (VOXEL_RIGID, VOXEL_SOFT, VOXEL_H_ACT, VOXEL_V_ACT):
            mask = (voxel_type_long == voxel_id) & valid
            if mask.any():
                token_values[mask] = self.value_decoders[voxel_id](decoder_input[mask]).squeeze(-1)

        denom = valid.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        value = (token_values * valid.float()).sum(dim=1, keepdim=True) / denom
        check_finite_tensor("critic_value", value)
        if return_hidden_states:
            return value, attention_maps, hidden_states
        return value, attention_maps
