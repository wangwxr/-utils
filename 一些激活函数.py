class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
swish = Swish()