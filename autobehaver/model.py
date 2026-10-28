"""Encode feature windows for center-frame PosePred classification and reconstruction."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .constants import TRANSFORMER_PARAM_INIT_STD

__all__ = ['PosePredTransformer']

class PosePredTransformer(nn.Module):
    """Encode temporal feature windows and apply the enabled output heads.

    Use the middle-frame, CLS-token or combined representation. Pretraining
    returns a projected embedding; supervised mode supports pose, orientation
    and optional sequence reconstruction.
    """
    def __init__(self, num_features: int, sequence_length: int, d_model: int, nhead: int, num_encoder_layers: int, num_decoder_layers: int, dim_feedforward: int, 
        dropout_rate: float, frame_representation: str, num_cls: int, prediction_head: str, pose_pred: bool, orient_pred: bool, recon: bool, 
        num_pose_classes: int, num_orient_classes: int, positional_encoding_type: str, pretrain_mode: bool = False, projection_dim: int = 128
    ) -> None:
        super(PosePredTransformer, self).__init__()
        self.d_model = d_model
        self.sequence_length = sequence_length
        self.frame_representation = frame_representation
        self.num_cls = num_cls
        self.pose_pred = pose_pred
        self.orient_pred = orient_pred
        self.recon = recon
        self.positional_encoding_type = positional_encoding_type
        self.pretrain_mode = pretrain_mode

        if prediction_head not in ['fc', 'mlp']:
            print(f"Invalid prediction_head '{prediction_head}' provided. Defaulting to 'fc'.")
            prediction_head = 'fc'
        self.prediction_head = prediction_head

        self.initial_linear = nn.Linear(num_features, d_model)
        
        if self.positional_encoding_type == 'learned':
            self.positional_encoding = nn.Parameter(torch.empty(1, sequence_length, d_model))
        elif self.positional_encoding_type == 'sinusoidal':
            pe = self._generate_sinusoidal_encoding(sequence_length, d_model)
            self.register_buffer('positional_encoding', pe)
        else:
            print(f"Warning: Unknown positional_encoding_type '{self.positional_encoding_type}'. Defaulting to 'learned'.")
            self.positional_encoding = nn.Parameter(torch.empty(1, sequence_length, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout_rate, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        if self.recon:            
            decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout_rate, batch_first=True)
            self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

            self.output_linear = nn.Linear(d_model, num_features)

        if frame_representation in ['cls', 'cls+middle']:
            self.frame_cls = nn.Parameter(torch.empty(1, num_cls, d_model))  
            self.frame_cls_positional_encoding = nn.Parameter(torch.empty(1, num_cls, d_model))  

        if frame_representation == 'cls+middle':
            self.concat_projection = nn.Linear((num_cls + 1) * d_model, d_model)

        elif frame_representation == 'cls':
            self.cls_projection = nn.Linear(num_cls * d_model, d_model)

        if self.pretrain_mode:
            in_features = self.d_model
            
            self.projection_head = nn.Sequential(
                nn.Linear(in_features, in_features),
                nn.BatchNorm1d(in_features),
                nn.ReLU(inplace=True),
                nn.Linear(in_features, projection_dim)
            )

        prediction_head_in_features = self.d_model

        if self.pose_pred:
            if self.prediction_head == 'fc':
                self.pose_fc = nn.Linear(prediction_head_in_features, num_pose_classes)
            elif self.prediction_head == 'mlp':
                self.pose_mlp = nn.Sequential(
                    nn.Linear(prediction_head_in_features, 4 * prediction_head_in_features),
                    nn.LayerNorm(4 * prediction_head_in_features),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(4 * prediction_head_in_features, 2 * prediction_head_in_features),
                    nn.LayerNorm(2 * prediction_head_in_features),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(2 * prediction_head_in_features, num_pose_classes)
                )

        if self.orient_pred:
            if self.prediction_head == 'fc':
                self.orient_fc = nn.Linear(prediction_head_in_features, num_orient_classes)
            elif self.prediction_head == 'mlp':
                self.orient_mlp = nn.Sequential(
                    nn.Linear(prediction_head_in_features, 4 * prediction_head_in_features),
                    nn.LayerNorm(4 * prediction_head_in_features),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(4 * prediction_head_in_features, 2 * prediction_head_in_features),
                    nn.LayerNorm(2 * prediction_head_in_features),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(2 * prediction_head_in_features, num_orient_classes)
                )

        # Apply the fixed transformer-style initialization policy after the full
        # module graph is constructed. This ensures cloned encoder/decoder layers
        # do not retain identical initial parameters from PyTorch's internal cloning.
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize matrices with Xavier uniform and biases with zero.

        Set normalization scales to one and biases to zero. Initialize learned
        token/position parameters from Normal(0, 0.02), preserving fixed sinusoidal
        encodings. Initialize after layer cloning so cloned layers differ.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                if getattr(module, 'in_proj_weight', None) is not None:
                    nn.init.xavier_uniform_(module.in_proj_weight)
                if getattr(module, 'in_proj_bias', None) is not None:
                    nn.init.zeros_(module.in_proj_bias)
            elif isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        if isinstance(getattr(self, 'positional_encoding', None), nn.Parameter):
            nn.init.normal_(self.positional_encoding, mean=0.0, std=TRANSFORMER_PARAM_INIT_STD)

        if isinstance(getattr(self, 'frame_cls', None), nn.Parameter):
            nn.init.normal_(self.frame_cls, mean=0.0, std=TRANSFORMER_PARAM_INIT_STD)

        if isinstance(getattr(self, 'frame_cls_positional_encoding', None), nn.Parameter):
            nn.init.normal_(self.frame_cls_positional_encoding, mean=0.0, std=TRANSFORMER_PARAM_INIT_STD)
            
    def _generate_sinusoidal_encoding(self, seq_len: int, d_model: int) -> torch.Tensor:
        """Return (1, T, D) sinusoidal encodings centered on the middle frame."""
        middle = seq_len // 2
        position = torch.arange(-middle, seq_len - middle, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, seq_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, src: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """Encode a (B, T, F) feature tensor.

        Return (pose_logits, orient_logits, reconstruction, context_vector), with
        None for disabled heads. Pretraining returns (None, None, None, embedding).
        """
        batch_size, sequence_length, num_features = src.shape
        
        src = self.initial_linear(src)
        
        src += self.positional_encoding[:, :self.sequence_length, :]
        
        src = src.view(batch_size, sequence_length, self.d_model)
        
        if self.frame_representation == 'cls' or self.frame_representation == 'cls+middle':
            frame_cls_expanded = self.frame_cls.expand(batch_size, self.num_cls, -1)
            frame_cls_positional_encoding_expanded = self.frame_cls_positional_encoding.expand(batch_size, self.num_cls, -1)
            src = torch.cat((frame_cls_expanded + frame_cls_positional_encoding_expanded, src), dim=1)
        
        memory = self.transformer_encoder(src)
            
        if self.frame_representation == 'middle':
            middle_index = self.sequence_length // 2
            middle_embedding = memory[:, middle_index, :]
            context_vector = middle_embedding
        elif self.frame_representation == 'cls':
            cls_embeddings = memory[:, :self.num_cls, :]
            concatenated_cls = cls_embeddings.reshape(-1, self.num_cls * self.d_model)
            context_vector = self.cls_projection(concatenated_cls)
        elif self.frame_representation == 'cls+middle':
            middle_index = self.num_cls + (self.sequence_length // 2)
            middle_embedding = memory[:, middle_index, :]
            cls_embeddings = memory[:, :self.num_cls, :]
            combined_embedding = torch.cat((cls_embeddings.reshape(-1, self.num_cls * self.d_model), middle_embedding), dim=-1)
            context_vector = self.concat_projection(combined_embedding)

        pose_pred = None
        orient_pred = None

        if self.pretrain_mode:
            context_vector = self.projection_head(context_vector)
            return None, None, None, context_vector

        if self.pose_pred:
            if self.prediction_head == 'fc':
                pose_pred = self.pose_fc(context_vector)
            elif self.prediction_head == 'mlp':
                pose_pred = self.pose_mlp(context_vector)

        if self.orient_pred:
            if self.prediction_head == 'fc':
                orient_pred = self.orient_fc(context_vector)
            elif self.prediction_head == 'mlp':
                orient_pred = self.orient_mlp(context_vector)
        
        if self.recon:
            # Strip the correct number of cls tokens before decoding
            if self.frame_representation == 'cls' or self.frame_representation == 'cls+middle':
                memory = memory[:, self.num_cls:, :]  
            
            recon_output = self.transformer_decoder(memory, memory)
            
            recon_output = self.output_linear(recon_output)
            
            recon_output = recon_output.view(batch_size, sequence_length, num_features)
        else:
            recon_output = None

        return pose_pred, orient_pred, recon_output, context_vector

