import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Assuming these modules are available in your project structure
from models.attention import SelfAttentionLayer, CrossAttentionLayer, FFNLayer
from models.position_embedding import PositionEmbeddingSine
from models.backbone import TransUNet
from models.utils import *

class UniOCTSeg_Plus(nn.Module):
    
    def __init__(self, task_num=8, hidden_dim=64, transformer_weights=None):
        super(UniOCTSeg_Plus, self).__init__()

        # Initialize Backbone (TransUNet)
        self.backbone = TransUNet()
        
        # Load pretrained transformer weights if provided
        if transformer_weights is not None:
            self.backbone.transformer.load_from(np.load(transformer_weights))

        # Define decoder channel configurations
        decoder_channels = [512, 256, 128, 64, 64, 64]

        # Initialize the Prompt Decoder
        self.Transformer_decoder = Prompt_decoder(
            in_channels=decoder_channels,
            center_channel=768,
            num_classes=task_num,
            hidden_dim=hidden_dim,
            num_queries=task_num,
            n_heads=8,
            dim_feedforward=int(hidden_dim * 8),
            dec_layers=10,
            pre_norm=False,
            mask_dim=hidden_dim,
            universal_flag=True
        )

    def forward(self, x, task_onehot):
        """
        Forward pass for UniOCTSeg_Plus.

        Args:
            x (torch.Tensor): Input image tensor.
            task_onehot (torch.Tensor): One-hot encoding or task-specific code.

        Returns:
            out_logits (torch.Tensor): Concatenated background and foreground logits.
        """
        # Extract features from the backbone
        masked_features, feature_list = self.backbone(x)
        
        # Pass features through the Transformer Decoder
        fore_out, back_out = self.Transformer_decoder(feature_list, masked_features, task_onehot)
        
        # Combine background and foreground outputs
        # Stacks them to form pairs [Background, Foreground] for each class/channel
        out_logits = torch.stack(
            [torch.cat([back_out[:, c:c+1], fore_out[:, c:c+1]], dim=1) 
             for c in range(fore_out.shape[1])], 
            dim=1
        )
        
        return out_logits


class Prompt_decoder(nn.Module):
    """
    Prompt Decoder Module.
    
    This module uses a Transformer decoder architecture with learnable queries
    and prompts to refine segmentation masks. It handles multi-scale features
    and produces specific foreground and background embeddings.
    """
    def __init__(self,
                 in_channels,
                 center_channel,
                 hidden_dim,
                 num_queries,
                 n_heads,
                 dim_feedforward,
                 dec_layers,
                 pre_norm,
                 mask_dim,
                 universal_flag=True
                 ):
        super().__init__()
        
        # --- Position Embedding ---
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        # --- Transformer Decoder Configuration ---
        self.num_heads = n_heads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        self.input_proj = nn.ModuleList()

        # Instantiate Transformer Layers
        for i in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=n_heads,
                    dropout=0.0,
                    normalize_before=pre_norm
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=n_heads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        # --- Projections ---
        # Projection layers to map input channels to hidden dimensions
        for i in range(len(in_channels)):
            self.input_proj.append(nn.Conv2d(in_channels[i], hidden_dim, kernel_size=1))
            
        self.mask_proj = nn.Conv2d(center_channel, hidden_dim, kernel_size=1)

        # Normalization Layers
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.fore_decoder_norm = nn.LayerNorm(hidden_dim)
        self.back_decoder_norm = nn.LayerNorm(hidden_dim)

        # --- Queries and Embeddings ---
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries + 1, hidden_dim)

        # Learnable query features
        self.query_feat = nn.ParameterList(
            [nn.Parameter(torch.randn(1, hidden_dim), requires_grad=True) for i in range(num_queries + 1)]
        )
        
        self.num_feature_levels = 5
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)

        # MLPs for mask generation
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        self.fore_mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        self.back_mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        # --- Task Specific Logic ---
        # Calculate task number based on universal flag (assuming sum_reverse is a util function)
        if universal_flag:
            task_num_calculated = sum_reverse(num_queries, 0)
        else:
            task_num_calculated = sum_reverse(num_queries, int(num_queries - 1))
            
        # Convolutional layers for processing task tokens
        self.foreground_token_list = nn.ModuleList(
            [nn.Conv1d(self.num_queries, 1, kernel_size=3, padding=1, stride=1) for _ in range(task_num_calculated)]
        )
        self.background_token_list = nn.ModuleList(
            [nn.Conv1d(self.num_queries, 1, kernel_size=3, padding=1, stride=1) for _ in range(task_num_calculated)]
        )

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        """
        Generates mask predictions and attention masks from decoder output.
        """
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)

        mask_embed = self.mask_embed(decoder_output)
        
        # Einstein summation to generate mask outputs
        # bqc: batch, query, channel
        # bchw: batch, channel, height, width
        # bqhw: batch, query, height, width
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # Upsample mask to target size for attention masking
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        
        # Convert to boolean mask for attention mechanisms
        # True values indicate positions that are NOT allowed to attend
        attn_mask = (
            attn_mask.sigmoid().flatten(2).unsqueeze(1)
            .repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5
        ).bool()
        
        attn_mask = attn_mask.detach()
        return outputs_mask, attn_mask, mask_embed
    
    def foreground_forward_prediction_heads(self, foreground, mask_features):
        """
        Specific prediction head for foreground masks.
        """
        decoder_foreground_output = self.fore_decoder_norm(foreground)
        decoder_foreground_output = decoder_foreground_output.transpose(0, 1)
        
        foreground_mask_embed = self.fore_mask_embed(decoder_foreground_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", foreground_mask_embed, mask_features)

        return outputs_mask
    
    def background_forward_prediction_heads(self, background, mask_features):
        """
        Specific prediction head for background masks.
        """
        decoder_background_output = self.back_decoder_norm(background)
        decoder_background_output = decoder_background_output.transpose(0, 1)
        
        background_mask_embed = self.back_mask_embed(decoder_background_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", background_mask_embed, mask_features)

        return outputs_mask

    def forward(self, x, mask_features, task_code, mask=None):
        """
        Forward pass for the Prompt Decoder.
        
        Args:
            x (list): List of multi-scale feature maps.
            mask_features (torch.Tensor): High-resolution features for masking.
            task_code (torch.Tensor): Tensor indicating task/class configuration.
        """
        mask_features = self.mask_proj(mask_features)
        
        # Prepare multi-scale features and position embeddings
        src = []
        pos = []
        size_list = []
        
        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            # Generate positional embeddings
            pos.append(self.pe_layer(x[i], None).flatten(2))
            # Project input features and add level embeddings
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # Flatten NxCxHxW to HWxNxC for Transformer
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape
        B, C, QN = task_code.shape

        # Expand query embeddings and features for the batch
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)
        
        # Concatenate query features for all queries + 1 (bias/null query)
        output = torch.cat(
            [self.query_feat[i] for i in range(self.num_queries + 1)], dim=0
        ).unsqueeze(1).repeat(1, B, 1)

        predictions_mask = []

        # Initial prediction to generate attention mask
        _, attn_mask, _ = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0])

        # --- Transformer Decoding Loop ---
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            
            # Reset attention mask where it is fully masked
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            
            # 1. Cross-Attention
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,
                pos=pos[level_index], query_pos=query_embed
            )
            
            # 2. Self-Attention
            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            
            # 3. Feed Forward Network
            output = self.transformer_ffn_layers[i](output)
            
            # Prepare features for the next prediction head
            current_mask_features = src[level_index].permute(1, 2, 0)
            current_mask_features = current_mask_features.view(
                x[level_index].shape[0], -1, x[level_index].shape[2], x[level_index].shape[3]
            )
            
            # Generate intermediate masks
            outputs_mask, attn_mask, _ = self.forward_prediction_heads(
                output, current_mask_features,
                attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels]
            )
            predictions_mask.append(outputs_mask)

        output = output.permute(1, 0, 2)
        
        # --- Task-Based Token Generation ---
        # Convert task code to indices
        task_index = task_onehot2task_index_universal_v2(task_code)
        
        # Generate Foreground Tokens
        # Selects specific query outputs based on the task_code and processes them via Conv1d
        foreground_list = torch.cat([
            torch.cat([
                self.foreground_token_list[task_index[i, j]](
                    torch.cat([
                        output[i:i+1, k+1:k+2] if task_code[i, j, k] != 0 else output[i:i+1, 0:1] 
                        for k in range(QN)
                    ], dim=1)
                ) for j in range(C)
            ], dim=1) for i in range(B)
        ], dim=0)

        # Generate Background Tokens
        # Similar logic but selects background-specific features (where task_code == 0)
        background_list = torch.cat([
            torch.cat([
                self.background_token_list[task_index[i, j]](
                    torch.cat([
                        output[i:i+1, k+1:k+2] if task_code[i, j, k] == 0 else output[i:i+1, 0:1] 
                        for k in range(QN)
                    ], dim=1)
                ) for j in range(C)
            ], dim=1) for i in range(B)
        ], dim=0)

        # Final permutation to match dimensions
        foreground_list = foreground_list.permute(1, 0, 2)
        background_list = background_list.permute(1, 0, 2)

        # Final prediction heads
        foreground_mask = self.foreground_forward_prediction_heads(foreground_list, x[-1])
        background_mask = self.background_forward_prediction_heads(background_list, x[-1])
        
        return foreground_mask, background_mask