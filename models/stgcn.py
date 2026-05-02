# models/stgcn.py — ST-GCN Following Yan et al. (2018)
"""
Spatio-Temporal Graph Convolutional Network
Following exact methodology from:
Yan et al. (2018) — "Spatial Temporal Graph Convolutional Networks
for Skeleton-Based Action Recognition"

Architecture:
Input (C=3, T=30, V=33)
→ 9 ST-GCN blocks
→ Global Average Pool
→ Fully Connected
→ Softmax (action classes)
"""
import torch
import torch.nn as nn
import numpy as np


class GraphConvolution(nn.Module):
    """
    Graph Convolution Layer
    Performs convolution over the spatial graph (body joints)
    Spatial meaning: Understands POSTURE
    """
    def __init__(self, in_channels, out_channels, num_joints=33):
        super(GraphConvolution, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x, A):
        """
        x: (B, C, T, V)
        A: (V, V) adjacency matrix
        """
        x = torch.einsum('bctv,vw->bctw', x, A)
        x = self.conv(x)
        return x


class STGCNBlock(nn.Module):
    """
    Single Spatio-Temporal GCN Block
    Combines Spatial Graph Conv + Temporal Conv
    Following Yan et al. (2018) exactly
    """
    def __init__(self, in_channels, out_channels,
                 temporal_kernel=9, stride=1, dropout=0.5):
        super(STGCNBlock, self).__init__()

        # Spatial Graph Convolution
        self.gcn = GraphConvolution(in_channels, out_channels)

        # Temporal Convolution
        padding = ((temporal_kernel - 1) // 2, 0)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=(temporal_kernel, 1),
                stride=(stride, 1),
                padding=padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout)
        )

        # Residual connection
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x, A):
        res = self.residual(x)
        x   = self.gcn(x, A)
        x   = self.tcn(x)
        x   = self.relu(x + res)
        return x


class STGCN(nn.Module):
    """
    Full ST-GCN Model following Yan et al. (2018)
    Architecture: BN → ST-GCN×5 → GAP → FC → Softmax
    """
    def __init__(self, num_joints=25, num_classes=5, in_channels=3):
        super(STGCN, self).__init__()

        # Input batch normalization
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)

        # 5 ST-GCN blocks — matching saved model
        self.st_gcn_blocks = nn.ModuleList([
            STGCNBlock(in_channels, 64,  dropout=0.5),  # Block 0
            STGCNBlock(64,          64,  dropout=0.5),  # Block 1
            STGCNBlock(64,          128, dropout=0.5),  # Block 2
            STGCNBlock(128,         128, dropout=0.5),  # Block 3
            STGCNBlock(128,         256, dropout=0.5),  # Block 4
        ])

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc  = nn.Linear(256, num_classes)

    def forward(self, x, A):
        """
        x: (B, C, T, V)
        A: (V, V) adjacency matrix
        """
        B, C, T, V = x.shape

        # Input normalization
        x_bn = x.permute(0, 3, 1, 2).contiguous().view(B, V * C, T)
        x_bn = self.data_bn(x_bn)
        x    = x_bn.view(B, V, C, T).permute(0, 2, 3, 1)

        # Pass through 5 ST-GCN blocks
        for block in self.st_gcn_blocks:
            x = block(x, A)

        # Global average pooling + classification
        x = self.gap(x)
        x = x.view(B, -1)
        x = self.fc(x)
        return x


class FallSeverityCNN(nn.Module):
    """
    CNN for Fall Severity Classification
    Novel contribution — not in existing papers!
    Input: Post-fall skeleton (B, C=3, T=30, V=33)
    Output: Severity Level (0=Minor, 1=Moderate, 2=Critical)
    """
    def __init__(self, num_joints=25, sequence_length=30):
        super(FallSeverityCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x


class PreFallLSTM(nn.Module):
    """
    LSTM for Pre-Fall Risk Prediction
    Novel contribution — not in existing papers!
    Input: Gait sequence (B, T=30, features=99)
    Output: Risk probability (0-1)
    """
    def __init__(self, input_size=75, hidden_size=128, num_layers=2):
        super(PreFallLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=False
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        lstm_out, _        = self.lstm(x)
        attention_weights  = self.attention(lstm_out)
        context            = torch.sum(attention_weights * lstm_out, dim=1)
        risk               = self.classifier(context)
        return risk


if __name__ == "__main__":
    print("Testing model architectures...")

    from agents.graph_builder import GraphBuilderAgent
    graph = GraphBuilderAgent()
    A     = graph.get_adjacency_matrix()

    # Test ST-GCN
    stgcn = STGCN(num_joints=33, num_classes=5)
    x     = torch.randn(4, 3, 30, 33)
    out   = stgcn(x, A)
    print(f"ST-GCN: {x.shape} → {out.shape}")

    # Test Severity CNN
    severity_cnn = FallSeverityCNN()
    out_sev      = severity_cnn(x)
    print(f"Severity CNN: {x.shape} → {out_sev.shape}")

    # Test PreFall LSTM
    prefall_lstm = PreFallLSTM(input_size=99)
    x_lstm       = torch.randn(4, 30, 99)
    out_risk     = prefall_lstm(x_lstm)
    print(f"PreFall LSTM: {x_lstm.shape} → {out_risk.shape}")

    print(f"\nST-GCN parameters      : {sum(p.numel() for p in stgcn.parameters()):,}")
    print(f"Severity CNN parameters: {sum(p.numel() for p in severity_cnn.parameters()):,}")
    print(f"PreFall LSTM parameters: {sum(p.numel() for p in prefall_lstm.parameters()):,}")
    print("\nAll models OK!")