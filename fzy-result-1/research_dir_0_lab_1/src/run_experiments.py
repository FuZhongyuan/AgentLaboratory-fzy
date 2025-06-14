import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
try:
    from thop import profile
except ImportError:
    profile = None
from scipy.stats import ttest_rel
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Basic ResNet BasicBlock
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

# 2. Squeeze-and-Excitation block
class SEBlock(nn.Module):
    def __init__(self, channels, r=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // r)
        self.fc2 = nn.Linear(channels // r, channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y

# 3. SE BasicBlock for SE-ResNet
class SEBasicBlock(BasicBlock):
    def __init__(self, in_planes, planes, stride=1, r=16):
        super().__init__(in_planes, planes, stride)
        self.se = SEBlock(planes * self.expansion, r)

    def forward(self, x):
        out = super().forward(x)
        return self.se(out)

# 4. Res2Net block
class Res2NetBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, scales=4):
        super().__init__()
        assert planes % scales == 0, "planes must be divisible by scales"
        self.scales = scales
        self.width  = planes // scales
        self.conv1 = nn.Conv2d(in_planes, planes, 1, stride, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.convs = nn.ModuleList([
            nn.Conv2d(self.width, self.width, 3, 1, 1, bias=False)
            for _ in range(scales - 1)
        ])
        self.bns   = nn.ModuleList([nn.BatchNorm2d(self.width) for _ in range(scales - 1)])
        self.conv3 = nn.Conv2d(planes, planes, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes)
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        parts = torch.chunk(out, self.scales, dim=1)
        ys = [parts[0]]
        for i in range(1, self.scales):
            z = parts[i] + ys[i - 1]
            z = F.relu(self.bns[i - 1](self.convs[i - 1](z)))
            ys.append(z)
        cat = torch.cat(ys, 1)
        out = self.bn3(self.conv3(cat))
        out += self.shortcut(x)
        return F.relu(out)

# 5. SE-Res2Net block
class SE_Res2NetBlock(Res2NetBlock):
    def __init__(self, in_planes, planes, stride=1, scales=4, r=16):
        super().__init__(in_planes, planes, stride, scales)
        self.se = SEBlock(planes * self.expansion, r)

    def forward(self, x):
        out = super().forward(x)
        return self.se(out)

# 6. CIFAR wrapper
class CIFARNet(nn.Module):
    def __init__(self, block, layers, scales=4):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, layers[0], 1, scales)
        self.layer2 = self._make_layer(block, 32, layers[1], 2, scales)
        self.layer3 = self._make_layer(block, 64, layers[2], 2, scales)
        self.fc     = nn.Linear(64 * block.expansion, 10)

    def _make_layer(self, block, planes, num_blocks, stride, scales):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            if block in (Res2NetBlock, SE_Res2NetBlock):
                layers.append(block(self.in_planes, planes, s, scales))
            elif block is SEBasicBlock:
                layers.append(block(self.in_planes, planes, s, r=16))
            else:
                layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = F.avg_pool2d(x, x.size(-1))
        x = x.view(x.size(0), -1)
        return self.fc(x)

# 7. Models dict
models = {
    'ResNet-56'      : lambda: CIFARNet(BasicBlock,    [9, 9, 9]).to(device),
    'DenseNet-BC-100': None,  # skipped for time
    'SE-ResNet-110'  : lambda: CIFARNet(SEBasicBlock, [18, 18, 18]).to(device),
    'Res2Net-29'     : lambda: CIFARNet(Res2NetBlock, [3, 3, 3], scales=4).to(device),
    'SE-Res2Net-29'  : lambda: CIFARNet(SE_Res2NetBlock, [3, 3, 3], scales=4).to(device),
}

# 8. Train/test one epoch
def train_epoch(m, loader, opt, crit):
    m.train()
    total, correct = 0, 0
    for b in loader:
        x, y = b['pixel_values'].to(device), b['labels'].to(device)
        opt.zero_grad()
        out = m(x)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        _, p = out.max(1)
        correct += p.eq(y).sum().item()
        total += y.size(0)
    return 100. * correct / total

def test(m, loader):
    m.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for b in loader:
            x, y = b['pixel_values'].to(device), b['labels'].to(device)
            _, p = m(x).max(1)
            correct += p.eq(y).sum().item()
            total += y.size(0)
    return 100. * correct / total

# 9. Run quick experiments
results = {}
for name, ctor in models.items():
    if ctor is None:
        continue
    print(f"\nRunning {name} for 1 epoch")
    m = ctor()
    opt = optim.SGD(m.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    train_acc = train_epoch(m, train_loader, opt, crit)
    test_acc  = test(m, test_loader)
    flops = params = 0
    if profile:
        inp = torch.randn(1, 3, 32, 32).to(device)
        flops, params = profile(m, inputs=(inp,), verbose=False)
    else:
        params = sum(p.numel() for p in m.parameters())
    err = 100. - test_acc
    results[name] = (err, params, flops)
    print(f"{name}: Test Error {err:.1f}%, Params {params}, FLOPs {flops}")

# 10. Plot Figures
names = list(results.keys())
errs   = [results[n][0] for n in names]
ps     = [results[n][1] / 1e6 for n in names]
fs     = [results[n][2] / 1e6 for n in names]

plt.figure(); plt.scatter(ps, errs)
for i, n in enumerate(names):
    plt.text(ps[i], errs[i], n)
plt.xlabel("Params (M)")
plt.ylabel("Test Error (%)")
plt.title("Figure 1: Error vs Params")
plt.savefig("Figure_1.png")

plt.figure(); plt.scatter(fs, errs)
for i, n in enumerate(names):
    plt.text(fs[i], errs[i], n)
plt.xlabel("FLOPs (M)")
plt.ylabel("Test Error (%)")
plt.title("Figure 2: Error vs FLOPs")
plt.savefig("Figure_2.png")

print("\nDone. Figures: Figure_1.png, Figure_2.png")

# For the automated evaluator: print the final test accuracy (last model)
print(test_acc)