import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, embed_dim)
        seq_len = x.size(1)
        x = x + self.pe[:seq_len].unsqueeze(0).to(x.device)
        return self.dropout(x)


class MotionPlannerTransformer(nn.Module):
    def __init__(
        self,
        lidar_dim: int,
        cmd_dim: int,
        goal_dim: int,
        action_dim: int,
        history_length: int,
        embed_dim: int,
        nhead: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        cnn_channels: list,
        cnn_kernel_size: int,
        num_pred_cmd: int
    ):
        '''
            complex implementation (with CNN encoder, larger params)
        '''
        super().__init__()
        self.history_length = history_length
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.num_pred_cmd = num_pred_cmd
        assert num_pred_cmd >= 1, "num_pred_cmd is the number of commands that will be predicted by the model. It includes the present command so it must be >= 1."


        # LiDAR CNN encoder
        cnn_layers = []
        in_ch = 1
        for out_ch in cnn_channels:
            cnn_layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=cnn_kernel_size, padding=cnn_kernel_size // 2))
            cnn_layers.append(nn.ReLU())
            in_ch = out_ch
        cnn_layers.append(nn.AdaptiveMaxPool1d(1))
        self.lidar_cnn = nn.Sequential(*cnn_layers)
        self.lidar_proj = nn.Linear(cnn_channels[-1], embed_dim)
        self.lidar_norm = nn.LayerNorm(embed_dim)
        self.cmd_projection = nn.Linear(cmd_dim, embed_dim)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(embed_dim, dropout, max_len=history_length+num_pred_cmd+5)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu' #could use relu for simplicity
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Goal MLP
        self.goal_mlp = nn.Sequential(
            nn.Linear(goal_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, embed_dim // 4)
        )

        # Output Head
        self.action_head = nn.Sequential(
            nn.Linear(embed_dim + embed_dim // 4, dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, action_dim * num_pred_cmd) #add *3 for predicting 3 actions ahead
        )
        '''
            quaternion layers (4 dimensions)
        '''
        # super().__init__()
        
        # if embed_dim % 4 != 0:
        #     raise ValueError(f"embed_dim ({embed_dim}) must be divisible by 4 for Quaternion layers.")
        # if dim_feedforward % 4 != 0:
        #     raise ValueError(f"dim_feedforward ({dim_feedforward}) must be divisible by 4 for Quaternion layers.")
        
        # self.history_length = history_length
        # self.embed_dim = embed_dim
        # self.action_dim = action_dim

        # cnn_layers = []
        # in_ch = 1
        # for out_ch in cnn_channels:
        #     cnn_layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=cnn_kernel_size, padding=cnn_kernel_size // 2))
        #     cnn_layers.append(nn.ReLU())
        #     in_ch = out_ch
        # cnn_layers.append(nn.AdaptiveMaxPool1d(1))
        # self.lidar_cnn = nn.Sequential(*cnn_layers)
        
        # final_cnn_channels = cnn_channels[-1]
        # if final_cnn_channels % 4 != 0:
        #     raise ValueError(f"The last channel size in cnn_channels ({final_cnn_channels}) must be divisible by 4.")
        
        # self.lidar_proj = QuaternionLinear(final_cnn_channels, embed_dim)
        
        # self.cmd_pre_proj = nn.Linear(cmd_dim, embed_dim) 
        # self.cmd_projection = QuaternionLinear(embed_dim, embed_dim)

        # self.lidar_norm = nn.LayerNorm(embed_dim)

        # self.pos_encoder = PositionalEncoding(embed_dim, dropout, max_len=history_length)

        # encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=embed_dim,
        #     nhead=nhead,
        #     dim_feedforward=dim_feedforward,
        #     dropout=dropout,
        #     batch_first=True,
        #     activation='gelu'
        # )
        # self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # self.goal_mlp = nn.Sequential(
        #     nn.Linear(goal_dim, embed_dim // 4),
        #     nn.ReLU(),
        #     nn.Linear(embed_dim // 4, embed_dim // 4)
        # )

        # ff_hidden_dim = (dim_feedforward // 2 // 4) * 4
        # embed_hidden_dim = (embed_dim // 2 // 4) * 4
        
        # self.action_head = nn.Sequential(
        #     QuaternionLinear(embed_dim + (embed_dim // 4), ff_hidden_dim),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     QuaternionLinear(ff_hidden_dim, embed_hidden_dim),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(embed_hidden_dim, action_dim)
        # )

    def forward(self, lidar_history: torch.Tensor, goal: torch.Tensor, arcmd: torch.Tensor) -> torch.Tensor:
        '''
            complex implementation (with CNN encoder, larger params) 
               - might try to add self-attention pooling for context vector
               - auto-regressive training??
        '''
        B, T, L = lidar_history.shape  # (batch, time, lidar_dim) batch first
        x = lidar_history.reshape(B * T, 1, L)
        x = self.lidar_cnn(x).squeeze(-1)
        x = self.lidar_proj(x).view(B, T, self.embed_dim)
        x = self.lidar_norm(x)
        if arcmd is not None:
            x_c = self.cmd_projection(arcmd) 
            x = x + x_c                               # (B, T, E)
        x = self.pos_encoder(x)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        x = self.transformer(x, mask=mask)      # (B, T, E)
        goal_seq = self.goal_mlp(goal).unsqueeze(1).expand(-1, T, -1)
        combined = torch.cat([x, goal_seq], dim=-1)
        action_preds = self.action_head(combined) 
        if self.num_pred_cmd > 1:
            action_preds = action_preds.view(B, T, self.num_pred_cmd, self.action_dim) #(B, T, 3, action_dim)
        
        return action_preds
        '''
            Quaternion
        '''
        # B, T, L = lidar_history.shape
        # x = lidar_history.reshape(B * T, 1, L)
        # x = self.lidar_cnn(x).squeeze(-1)
        # x = self.lidar_proj(x).view(B, T, self.embed_dim)
        # x = self.lidar_norm(x)

        # if arcmd is not None:
        #     x_c = self.cmd_pre_proj(arcmd)
        #     x_c = self.cmd_projection(x_c)
        #     x = x + x_c

        # x = self.pos_encoder(x)
        # mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        # x = self.transformer(x, mask=mask)
        # goal_seq = self.goal_mlp(goal).unsqueeze(1).expand(-1, T, -1)
        # combined = torch.cat([x, goal_seq], dim=-1)
        

        # combined_features = combined.shape[-1]
        # if combined_features % 4 != 0:
        #     padding_size = 4 - (combined_features % 4)
        #     padding = torch.zeros(B, T, padding_size, device=combined.device)
        #     combined = torch.cat([combined, padding], dim=-1)

        # action_preds = self.action_head(combined)
        
        # return action_preds