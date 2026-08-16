import torch
import math
import torch
import torch.nn as nn
from .transformer import TransformerEncoder
from .transformer import TransformerEncoderLayerResidual

def make_mlp_default(dim_list, final_nonlinearity=True, nonlinearity="relu"):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if nonlinearity == "relu":
            layers.append(nn.ReLU())
        elif nonlinearity == "tanh":
            layers.append(nn.Tanh())

    if not final_nonlinearity:
        layers.pop()
    return nn.Sequential(*layers)
    
class TransformerModel(nn.Module):
    # feature_size: dimension of input feature size
    # output_size: action dimension (normally 1)
    # ninp: dimension of Projection
    # nhead: number of attention head
    # nhid: hidden layer size of transformer encoder
    # nlayers: number of hidden layers of transformer encoder
    # condition_decoder: use input as query to inform the output
    # transformer_norm: layer_normalization
    # sequence_size: Robot size: e.g. 5x5
    # other_feature_size : other_feature_size
    def __init__(
        self,
        feature_size,
        output_size,
        sequence_size,
        other_feature_size,
        ninp,
        nhead,
        nhid,
        nlayers,
        dropout=0.5,
        args=None,
        use_transformer=None,
        is_actor = True
    ):
        """This model is built upon https://pytorch.org/tutorials/beginner/transformer_tutorial.html"""
        super(TransformerModel, self).__init__()
        self.args=args
        self.model_type = "Transformer"
        self.seq_len = sequence_size
        self.decoder_input_dim = ninp
        self.ninp = ninp
        self.use_transformer = True if use_transformer == 'transformer' else False
        self.is_actor = is_actor

        if self.use_transformer:
            # Position embedding
            if self.args.POS_EMBEDDING == "learnt":
                self.pos_embedding = PositionalEncoding(ninp, self.seq_len)
            elif self.args.POS_EMBEDDING == "abs":
                self.pos_embedding = PositionalEncoding1D(ninp, self.seq_len)
            elif self.args.POS_EMBEDDING == "abs2":
                self.pos_embedding = PositionalEncoding2D(ninp, self.seq_len)
            elif self.args.POS_EMBEDDING == "learnt2c":
                self.pos_embedding = PositionalEncoding2C(ninp, self.seq_len)
            elif self.args.POS_EMBEDDING == "learnt2_c":
                self.pos_embedding = PositionalEncoding2_C(ninp, self.seq_len)
            elif self.args.POS_EMBEDDING == "coord":
                ninp-=2
        
            # Transformer Encoder
            encoder_layers = TransformerEncoderLayerResidual(self.ninp, nhead, nhid, dropout)

            self.transformer_encoder = TransformerEncoder(
                encoder_layers,
                nlayers,
                norm=nn.LayerNorm(self.ninp) if self.args.transformer_norm else None,
            )

        # Linear Projection for input features
        if self.args.use_separate_pos_embedding:
            self.encoderx = nn.Linear(feature_size//2, ninp//2)
            self.encodery = nn.Linear(feature_size//2, ninp//2)
        else:
            self.encoder = nn.Linear(feature_size, ninp)

        self.condition_decoder = self.args.condition_decoder
        # decoder
        if self.args.use_other_obs_encoder:
            hidden_dim = [32, other_feature_size]
            # Task-related observation encoder  
            self.other_info_encoder = MLPObsEncoder(other_feature_size, hidden_dim)
            self.decoder_input_dim += self.other_info_encoder.obs_feat_dim
        else:
            self.decoder_input_dim += other_feature_size

        if self.condition_decoder:
            self.decoder_input_dim += feature_size

        if not self.is_actor:
            self.decoder = make_mlp_default([self.decoder_input_dim] + [64] + [output_size],
                final_nonlinearity=True, nonlinearity='relu')
        else:
            self.decoder = make_mlp_default([self.decoder_input_dim] + [64] + [output_size],
                final_nonlinearity=True, nonlinearity='tanh')

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        if self.args.use_separate_pos_embedding:
            self.encoderx.weight.data.uniform_(-initrange, initrange)
            self.encodery.weight.data.uniform_(-initrange, initrange)
        else:
            self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder[-2].bias.data.zero_()
        self.decoder[-2].weight.data.uniform_(-initrange, initrange)

    def forward(self, modular_state, other_state, obs_mask, obs_coord,
                attn_mask=None, return_attn=False, return_hidden_states=False):
        
        # Linear Porjection of local observation

        # (batch_size,num_modular, feature_size)

        if self.args.use_separate_pos_embedding:
            modular_state_x, modular_state_y = torch.chunk(modular_state, chunks=2, dim=-1)
            obs_embed_x = self.encoderx(modular_state_x)
            obs_embed_y = self.encodery(modular_state_y)
            obs_embed = torch.cat([obs_embed_x,obs_embed_y],axis=2)* math.sqrt(self.ninp)
        else:
            # (batch_size,num_modular, ninp)
            obs_embed = self.encoder(modular_state) * math.sqrt(self.ninp)

        _,batch_size, _ = obs_embed.shape
        
        # Linear porjection of other observation

        if self.args.use_other_obs_encoder:
            # (batch_size, embed_size)
            other_obs_embed = self.other_info_encoder(other_state)
        else:
            # (batch_size, embed_size)
            other_obs_embed = other_state

        # (batch_size, embed_size x self.seq_len/num_modular)
        other_obs_embed = other_obs_embed.repeat(self.seq_len, 1)
        # (self.seq_len/num_modular, batch_size, other_feature_size)
        other_obs_embed = other_obs_embed.reshape(self.seq_len,batch_size, -1)

        if self.use_transformer:
            # Position embedding
            if self.args.POS_EMBEDDING == "coord":
                obs_coord=obs_coord.repeat(batch_size,1)
                obs_coord=obs_coord.reshape(self.seq_len,batch_size, -1)
                obs_embed = torch.cat([obs_embed,obs_coord],axis=2)
            else:
                obs_embed = self.pos_embedding(obs_embed)
            if return_attn and return_hidden_states:
                obs_embed_t, attn_weights, hidden_states = self.transformer_encoder(
                    obs_embed,
                    mask=attn_mask,
                    src_key_padding_mask=obs_mask,
                    return_attn=True,
                    return_hidden_states=True,
                )
            elif return_attn:
                obs_embed_t, attn_weights = self.transformer_encoder(
                    obs_embed,
                    mask=attn_mask,
                    src_key_padding_mask=obs_mask,
                    return_attn=True,
                )
            elif return_hidden_states:
                obs_embed_t, hidden_states = self.transformer_encoder(
                    obs_embed,
                    mask=attn_mask,
                    src_key_padding_mask=obs_mask,
                    return_hidden_states=True,
                )
            else:
                obs_embed_t = self.transformer_encoder(
                    obs_embed, mask=attn_mask, src_key_padding_mask=obs_mask
                )
            decoder_input = obs_embed_t
        else:
            decoder_input = obs_embed

        if self.condition_decoder:
            decoder_input = torch.cat([decoder_input, other_obs_embed, modular_state], axis=2)
        else:
            decoder_input = torch.cat([decoder_input, other_obs_embed], axis=2)
        
        output = self.decoder(decoder_input)
        output = output.permute(1,0,2)
        output = output.reshape(batch_size,-1)
        # output = output.reshape(batch_size,num_modular,-1)
        if return_attn and return_hidden_states:
            return output, attn_weights, hidden_states
        if return_attn:
            return output, attn_weights
        if return_hidden_states:
            return output, hidden_states
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(seq_len,1,d_model))

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe
        return self.dropout(x)
    
class PositionalEncoding2D(nn.Module):

    def __init__(self, d_model, seq_len, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        h = int(math.sqrt(seq_len))
        w = int(math.sqrt(seq_len))
        d_model_half = d_model // 2
        
        div_term = torch.exp(torch.arange(0, d_model_half, 2) * (-math.log(10000.0) / d_model_half))
        
        pe = torch.zeros(h, w, d_model)
        
        y_pos = torch.arange(h).unsqueeze(1)
        x_pos = torch.arange(w).unsqueeze(1)
        
        pe[:, :, 0:d_model_half:2] = torch.sin(y_pos * div_term).unsqueeze(1).repeat(1, w, 1)
        pe[:, :, 1:d_model_half:2] = torch.cos(y_pos * div_term).unsqueeze(1).repeat(1, w, 1)
        pe[:, :, d_model_half::2] = torch.sin(x_pos * div_term).unsqueeze(0).repeat(h, 1, 1)
        pe[:, :, d_model_half+1::2] = torch.cos(x_pos * div_term).unsqueeze(0).repeat(h, 1, 1)
        
        pe = pe.view(seq_len,1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe
        return self.dropout(x)

class PositionalEncoding2_C(nn.Module):
    def __init__(self, d_model, seq_len, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.h = int(math.sqrt(seq_len))
        self.w = int(math.sqrt(seq_len))

        self.x_embed = nn.Embedding(self.h, d_model//2)
        self.y_embed = nn.Embedding(self.w, d_model//2)

        
        pos_x = torch.arange(self.h).unsqueeze(1).repeat(1, self.w).reshape(-1)  
        pos_y = torch.arange(self.w).unsqueeze(0).repeat(self.h, 1).reshape(-1) 
        self.register_buffer("pos_x", pos_x, persistent=False)
        self.register_buffer("pos_y", pos_y, persistent=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.x_embed.weight, -0.1, 0.1)
        nn.init.uniform_(self.y_embed.weight, -0.1, 0.1)

    def forward(self, x):
        """
        x: [L,B,D]
        """
        pe =torch.cat([self.x_embed(self.pos_x), self.y_embed(self.pos_y)], axis=1)
        pe = pe.to(dtype=x.dtype)
        x = x + pe.unsqueeze(1)
        return self.dropout(x)

class PositionalEncoding2C(nn.Module):
    def __init__(self, d_model, seq_len, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.h = int(math.sqrt(seq_len))
        self.w = int(math.sqrt(seq_len))

        self.x_embed = nn.Embedding(self.h, d_model)  
        self.y_embed = nn.Embedding(self.w, d_model)  

        
        pos_x = torch.arange(self.h).unsqueeze(1).repeat(1, self.w).reshape(-1)  # [L]
        pos_y = torch.arange(self.w).unsqueeze(0).repeat(self.h, 1).reshape(-1)  # [L]
        self.register_buffer("pos_x", pos_x, persistent=False)
        self.register_buffer("pos_y", pos_y, persistent=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.x_embed.weight, -0.1, 0.1)
        nn.init.uniform_(self.y_embed.weight, -0.1, 0.1)

    def forward(self, x):
        """
        x: [L,B,D]
        """
        pe = self.x_embed(self.pos_x) + self.y_embed(self.pos_y)
        pe = pe.to(dtype=x.dtype)
        x = x + pe.unsqueeze(1)
        return self.dropout(x)

class PositionalEncoding1D(nn.Module):

    def __init__(self, d_model, seq_len, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe
        return self.dropout(x)

class MLPObsEncoder(nn.Module):
    """Encoder for other env obs."""

    def __init__(self, obs_dim,hidden_size):
        super(MLPObsEncoder, self).__init__()
        mlp_dims = [obs_dim] + hidden_size
        self.encoder = make_mlp_default(mlp_dims)
        self.obs_feat_dim = mlp_dims[-1]

    def forward(self, obs):
        return self.encoder(obs)
