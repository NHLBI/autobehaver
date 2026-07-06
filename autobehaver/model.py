"""Model definitions for Autobehaver pose/orientation prediction.

The core network is a Transformer encoder (optionally with a decoder/reconstruction head) that
produces a per-sequence context representation used to predict Pose and Orientation labels for
the center frame of the input window."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .constants import TRANSFORMER_PARAM_INIT_STD

__all__ = ['PosePredTransformer']

class PosePredTransformer(nn.Module):
    """
    Multi-task transformer for sequence classification and reconstruction.
    
    Supports:
    - Pose classification, orientation prediction, sequence reconstruction
    - Multiple frame representations (cls tokens, middle frame, combined)
    - Pretraining mode with contrastive learning projection head
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

        # Validate prediction_head parameter
        if prediction_head not in ['fc', 'mlp']:
            print(f"Invalid prediction_head '{prediction_head}' provided. Defaulting to 'fc'.")
            prediction_head = 'fc'
        self.prediction_head = prediction_head

        # Initial Linear Layer: Transforms the input feature space to match the Transformer model's expected input dimension (d_model)
        self.initial_linear = nn.Linear(num_features, d_model)
        
        # Positional Encoding: Choose between learned and sinusoidal
        if self.positional_encoding_type == 'learned':
            self.positional_encoding = nn.Parameter(torch.empty(1, sequence_length, d_model))
        elif self.positional_encoding_type == 'sinusoidal':
            pe = self._generate_sinusoidal_encoding(sequence_length, d_model)
            self.register_buffer('positional_encoding', pe)
        else:
            print(f"Warning: Unknown positional_encoding_type '{self.positional_encoding_type}'. Defaulting to 'learned'.")
            self.positional_encoding = nn.Parameter(torch.empty(1, sequence_length, d_model))
        
        # Encoder: Composed of a stack of N encoder layers that process the input sequence
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout_rate, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        if self.recon:            
            # Decoder: Composed of a stack of N decoder layers that generate the output sequence from the context vector
            decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout_rate, batch_first=True)
            self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

            # Output Linear Layer: Maps the Transformer output back to the original feature space
            self.output_linear = nn.Linear(d_model, num_features)

        # Add CLS tokens if using 'cls' or 'cls+middle' frame representation
        if frame_representation in ['cls', 'cls+middle']:
            # Create num_cls learnable frame class tokens
            self.frame_cls = nn.Parameter(torch.empty(1, num_cls, d_model))  # Learnable frame class tokens
            self.frame_cls_positional_encoding = nn.Parameter(torch.empty(1, num_cls, d_model))  # Positional encoding for frame class tokens

        # Linear layer to project concatenated embeddings back to d_model
        if frame_representation == 'cls+middle':
            # Adjust the projection layer to account for multiple CLS tokens
            self.concat_projection = nn.Linear((num_cls + 1) * d_model, d_model)

        # Add a projection layer for 'cls' representation if using multiple CLS tokens
        elif frame_representation == 'cls':
            self.cls_projection = nn.Linear(num_cls * d_model, d_model)

        # In pretraining mode, add a projection head
        if self.pretrain_mode:
            # The context vector is d_model after projection
            in_features = self.d_model
            
            self.projection_head = nn.Sequential(
                nn.Linear(in_features, in_features),
                nn.BatchNorm1d(in_features),
                nn.ReLU(inplace=True),
                nn.Linear(in_features, projection_dim)
            )

        # Initialize classification heads if needed
        # Prediction heads receive d_model features
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
        """Initialize the full model with a fixed transformer-style policy.

        Rules:
        - Xavier uniform for all projection/MLP matrix weights.
        - Zero all biases.
        - LayerNorm / BatchNorm scales to one and biases to zero.
        - Learned token/positional parameters from Normal(0, 0.02).
        - Fixed sinusoidal positional encodings are left untouched.
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
        """Creates sinusoidal positional encoding for sequence positions, centered on the middle."""
        middle = seq_len // 2
        position = torch.arange(-middle, seq_len - middle, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, seq_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, src: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass through transformer encoder with multi-task outputs.
        
        Returns: (pose_logits, orient_logits, reconstruction, context_vector)
        In pretraining mode: (None, None, None, projected_embedding)
        """
        batch_size, sequence_length, num_features = src.shape
        
        # Step 1: Transform input feature space to d_model dimension
        src = self.initial_linear(src)
        
        # Step 2: Add positional encoding (and CLS tokens if applicable)
        src += self.positional_encoding[:, :self.sequence_length, :]
        
        # Reshape src to 3D for concatenation
        src = src.view(batch_size, sequence_length, self.d_model)
        
        if self.frame_representation == 'cls' or self.frame_representation == 'cls+middle':
            # Expand frame_cls token and its positional encoding to match batch size and num_cls
            frame_cls_expanded = self.frame_cls.expand(batch_size, self.num_cls, -1)
            frame_cls_positional_encoding_expanded = self.frame_cls_positional_encoding.expand(batch_size, self.num_cls, -1)
            # Concatenate along the sequence dimension
            src = torch.cat((frame_cls_expanded + frame_cls_positional_encoding_expanded, src), dim=1)
        
        # Step 3: Encode the sequence
        memory = self.transformer_encoder(src)
            
        # Step 4: Frame representation logic for context vector generation
        if self.frame_representation == 'middle':
            middle_index = self.sequence_length // 2
            middle_embedding = memory[:, middle_index, :]
            # Use middle_embedding as context_vector
            context_vector = middle_embedding
        elif self.frame_representation == 'cls':
            # Extract the first num_cls tokens
            cls_embeddings = memory[:, :self.num_cls, :]
            # Concatenate the CLS tokens to form the context vector
            concatenated_cls = cls_embeddings.reshape(-1, self.num_cls * self.d_model)
            # Project back to d_model
            context_vector = self.cls_projection(concatenated_cls)
        elif self.frame_representation == 'cls+middle':
            # Adjust middle_index to account for the cls tokens
            middle_index = self.num_cls + (self.sequence_length // 2)
            middle_embedding = memory[:, middle_index, :]
            # Extract the first num_cls tokens
            cls_embeddings = memory[:, :self.num_cls, :]
            # Concatenate cls and middle embeddings
            combined_embedding = torch.cat((cls_embeddings.reshape(-1, self.num_cls * self.d_model), middle_embedding), dim=-1)
            # Project back to d_model
            context_vector = self.concat_projection(combined_embedding)

        # Step 5: Process context vector depending on the task(s)
        # Initialize predictions
        pose_pred = None
        orient_pred = None

        # If in pretraining mode, pass the context vector through the projection head
        if self.pretrain_mode:
            context_vector = self.projection_head(context_vector)
            return None, None, None, context_vector

        # Pose prediction
        if self.pose_pred:
            if self.prediction_head == 'fc':
                pose_pred = self.pose_fc(context_vector)
            elif self.prediction_head == 'mlp':
                pose_pred = self.pose_mlp(context_vector)

        # Orientation prediction
        if self.orient_pred:
            if self.prediction_head == 'fc':
                orient_pred = self.orient_fc(context_vector)
            elif self.prediction_head == 'mlp':
                orient_pred = self.orient_mlp(context_vector)
        
        # Step 6: Reconstruction (if enabled)
        if self.recon:
            # Strip the correct number of cls tokens before decoding
            if self.frame_representation == 'cls' or self.frame_representation == 'cls+middle':
                memory = memory[:, self.num_cls:, :]  # Remove the first num_cls tokens (cls tokens)
            
            # Decode the memory
            recon_output = self.transformer_decoder(memory, memory)
            
            # Map decoder output back to original feature space
            recon_output = self.output_linear(recon_output)
            
            # Reshape output
            recon_output = recon_output.view(batch_size, sequence_length, num_features)
        else:
            recon_output = None

        return pose_pred, orient_pred, recon_output, context_vector

