"""Multi-head CNN with squeeze-excitation and TEMPORAL attention pooling (attention is defined over input messages)."""
import torch, torch.nn as nn, torch.nn.functional as F


class SE(nn.Module):
    def __init__(s, c, r=4):
        super().__init__(); s.f1, s.f2 = nn.Linear(c, max(c // r, 4)), nn.Linear(max(c // r, 4), c)
    def forward(s, x): return x * torch.sigmoid(s.f2(F.relu(s.f1(x.mean(-1)))))[..., None]


class Head(nn.Module):
    def __init__(s, f_in, ch, k):
        super().__init__()
        s.c1, s.bn, s.se = nn.Conv1d(f_in, ch, k, padding=k // 2), nn.BatchNorm1d(ch), SE(ch)
        s.c2, s.score = nn.Conv1d(ch, ch, k, padding=k // 2), nn.Conv1d(ch, 1, 1)
    def forward(s, x):
        h = F.relu(s.c2(s.se(F.relu(s.bn(s.c1(x))))))
        a = torch.softmax(s.score(h).squeeze(1), -1)                  # (B,T)
        return (h * a.unsqueeze(1)).sum(-1), a


class AIMHCNN(nn.Module):
    def __init__(s, n_out, f_in, ch=48, kernels=(3, 5, 7), emb=128, drop=0.3):
        super().__init__()
        s.heads = nn.ModuleList(Head(f_in, ch, k) for k in kernels)
        s.fc, s.drop, s.cls = nn.Linear(ch * len(kernels), emb), nn.Dropout(drop), nn.Linear(emb, n_out)
    def forward(s, x):                                                # x: (B,T,F) -> logits, embedding, attention (B,H,T)
        x = x.transpose(1, 2); o = [h(x) for h in s.heads]
        z = F.relu(s.fc(torch.cat([a for a, _ in o], 1)))
        return s.cls(s.drop(z)), z, torch.stack([b for _, b in o], 1)
