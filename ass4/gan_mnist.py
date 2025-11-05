import os, random, math, argparse, time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils
import matplotlib.pyplot as plt


def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Generator(nn.Module):
    def __init__(self, z_dim=100, img_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256 * 7 * 7),
            nn.BatchNorm1d(256 * 7 * 7),
            nn.ReLU(True),
            nn.Unflatten(1, (256, 7, 7)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, img_channels, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z): return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, img_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1)  # logits
        )

    def forward(self, x): return self.net(x).squeeze(1)


def label_smoothing(target_is_real, size, device):
    if target_is_real:
        return torch.empty(size, device=device).uniform_(0.8, 1.0)
    else:
        return torch.empty(size, device=device).uniform_(0.0, 0.2)


def get_data(root, batch_size, num_workers=2, pin_memory=False):
    tfm = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    ds = datasets.MNIST(root=root, train=True, download=True, transform=tfm)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


def save_grid(tensor, path, nrow=8):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    grid = vutils.make_grid(tensor, nrow=nrow, normalize=True, value_range=(-1, 1))
    vutils.save_image(grid, path)


def plot_curves(hist, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure()
    plt.plot(hist['g'], label='G')
    plt.plot(hist['d'], label='D')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def train(args):
    seed_everything(args.seed)
    if torch.backends.mps.is_available() and not args.cpu:
        device = torch.device("mps")
    elif torch.cuda.is_available() and not args.cpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if device.type == "mps":
        print("Using Apple MPS (Metal) on Apple Silicon")
    elif device.type == "cuda":
        print("Using NVIDIA CUDA:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU")

    pin_mem = (device.type == "cuda")
    dl = get_data(args.data, args.batch_size, args.workers, pin_memory=pin_mem)
    G = Generator(args.z_dim, img_channels=1).to(device)
    D = Discriminator(img_channels=1).to(device)
    for m in (G, D):
        for p in m.parameters():
            if p.dim() > 1: nn.init.kaiming_normal_(p)

    g_opt = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=args.wd)
    d_opt = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=args.wd)
    g_sch = torch.optim.lr_scheduler.StepLR(g_opt, step_size=max(1, args.lr_step), gamma=args.lr_gamma)
    d_sch = torch.optim.lr_scheduler.StepLR(d_opt, step_size=max(1, args.lr_step), gamma=args.lr_gamma)

    loss_fn = nn.BCEWithLogitsLoss()
    fix_z = torch.randn(args.sample_rows * args.sample_cols, args.z_dim, device=device)
    hist = {'g': [], 'd': []}
    it = 0

    for epoch in range(1, args.epochs + 1):
        G.train()
        D.train()
        for x, _ in dl:
            x = x.to(device, non_blocking=True)

            # Train D
            d_opt.zero_grad(set_to_none=True)
            z = torch.randn(x.size(0), args.z_dim, device=device)
            with torch.no_grad():
                fake = G(z)
            real_logits = D(x)
            fake_logits = D(fake)
            real_targets = label_smoothing(True, real_logits.shape, device)
            fake_targets = label_smoothing(False, fake_logits.shape, device)
            d_loss = loss_fn(real_logits, real_targets) + loss_fn(fake_logits, fake_targets)
            d_loss.backward()
            if args.grad_clip > 0: nn.utils.clip_grad_norm_(D.parameters(), args.grad_clip)
            d_opt.step()

            # Train G
            g_opt.zero_grad(set_to_none=True)
            z = torch.randn(x.size(0), args.z_dim, device=device)
            gen = G(z)
            logits = D(gen)
            g_targets = torch.ones_like(logits, device=device) * 0.95  # mild smoothing
            g_loss = loss_fn(logits, g_targets)
            g_loss.backward()
            if args.grad_clip > 0: nn.utils.clip_grad_norm_(G.parameters(), args.grad_clip)
            g_opt.step()

            hist['g'].append(g_loss.item());
            hist['d'].append(d_loss.item())
            it += 1

        g_sch.step();
        d_sch.step()

        with torch.no_grad():
            G.eval()
            samples = G(fix_z).cpu()
        save_grid(samples, os.path.join(args.out, f'samples_epoch_{epoch:03d}.png'), nrow=args.sample_cols)

        print(
            f'Epoch {epoch}/{args.epochs} | D: {d_loss.item():.3f} | G: {g_loss.item():.3f} | lr: {g_opt.param_groups[0]["lr"]:.5f}')

    plot_curves(hist, os.path.join(args.out, 'training_curves.png'))

    # Brief analysis (stdout)
    print("\\n=== Brief Analysis ===")
    print("• Check samples_epoch_*.png for progression in sharpness/diversity.")
    print(
        "• Convergence: Expect D and G losses to fluctuate; sustained collapse suggests lr too high or insufficient capacity.")
    print("• Stabilisation used: label smoothing, BatchNorm, Adam (β1=0.5), StepLR, grad clipping.")
    print(
        "• Mode collapse hints: repeated digits with identical strokes; mitigate via lower lr, noise on D inputs, or data aug.")
    print("• If training is slow on CPU, reduce epochs or raise batch size to improve throughput.")
    print("======================\\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, default='./data')
    p.add_argument('--out', type=str, default='./results')
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--z_dim', type=int, default=100)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--wd', type=float, default=0.0)
    p.add_argument('--lr_step', type=int, default=10)
    p.add_argument('--lr_gamma', type=float, default=0.5)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--sample_rows', type=int, default=8)
    p.add_argument('--sample_cols', type=int, default=8)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    train(args)


if __name__ == '__main__':
    main()
