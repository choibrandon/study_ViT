#1월 21일 mlp 구성
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer=nn.GELU, bias=True, drop=0):
        super().__init__()
        out_features = in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

if __name__ == '__main__':
    def main():
        mlp_layer = MLP(in_features=192, hidden_features=384)
        sample_input = torch.randn(1, 10, 192)
        output = mlp_layer(sample_input)
        print("MLP output shape:", output.shape)
    main()