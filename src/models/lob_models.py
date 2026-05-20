import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepLOB(nn.Module):
    def __init__(self, num_classes=3, num_horizons=5):
        super(DeepLOB, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(1, 2), stride=(1, 2))
        self.conv2 = nn.Conv2d(16, 16, kernel_size=(4, 1))
        self.conv3 = nn.Conv2d(16, 16, kernel_size=(4, 1))
        
        self.fc_temporal = nn.Linear(320, 64) 
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True)
        self.heads = nn.ModuleList([nn.Linear(64, num_classes) for _ in range(num_horizons)])

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        batch_size, channels, T_prime, feats = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, T_prime, -1)
        
        x = F.relu(self.fc_temporal(x))
        lstm_out, _ = self.lstm(x)
        shared_features = lstm_out[:, -1, :]
        return [head(shared_features) for head in self.heads]

class LOBCAST(nn.Module):
    def __init__(self, num_classes=3, num_horizons=5):
        super(LOBCAST, self).__init__()
        self.causal_conv1 = nn.Conv1d(in_channels=40, out_channels=32, kernel_size=3, padding=2, dilation=1)
        self.causal_conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=4, dilation=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.heads = nn.ModuleList([nn.Linear(64, num_classes) for _ in range(num_horizons)])

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.causal_conv1(x))[:, :, :-2] 
        x = F.relu(self.causal_conv2(x))[:, :, :-4] 
        shared_features = self.pool(x).squeeze(-1)
        return [head(shared_features) for head in self.heads]

class LOBTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2, num_classes=3, num_horizons=5, T=100):
        super(LOBTransformer, self).__init__()
        self.input_projection = nn.Linear(40, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, T, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, activation='relu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.heads = nn.ModuleList([nn.Linear(d_model, num_classes) for _ in range(num_horizons)])

    def forward(self, x):
        x = self.input_projection(x) + self.pos_embedding
        mask = torch.triu(torch.full((x.size(1), x.size(1)), float('-inf'), device=x.device), diagonal=1)
        x = self.transformer(x, mask=mask)
        shared_features = x[:, -1, :]
        return [head(shared_features) for head in self.heads]

class MambaLOB(nn.Module):
    def __init__(self, d_model=64, d_state=16, num_classes=3, num_horizons=5):
        super(MambaLOB, self).__init__()
        self.projection = nn.Linear(40, d_model)
        self.A_param = nn.Parameter(torch.randn(d_model, d_state))
        self.B_projection = nn.Linear(d_model, d_state)
        self.C_projection = nn.Linear(d_model, d_state)
        self.heads = nn.ModuleList([nn.Linear(d_model, num_classes) for _ in range(num_horizons)])

    def forward(self, x):
        x = self.projection(x)
        batch_size, T, d_model = x.size()
        hidden_state = torch.zeros(batch_size, d_model, self.A_param.size(1), device=x.device)
        
        for t in range(T):
            step_x = x[:, t, :]
            B = self.B_projection(step_x).unsqueeze(1)
            hidden_state = hidden_state * torch.sigmoid(self.A_param) + torch.bmm(step_x.unsqueeze(2), B)
            
        shared_features = torch.sum(hidden_state, dim=-1)
        return [head(shared_features) for head in self.heads]
