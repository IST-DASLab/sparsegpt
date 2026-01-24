import math
import time

import torch
import torch.nn as nn
import transformers

from quant import *


DEBUG = False 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def venom_patterns_2of4(device):
    # True = prune, False = keep
    patterns = torch.tensor([
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 1],
    ], dtype=torch.bool, device=device)
    return patterns

class SparseGPT:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out, blocksize=1024):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterprune(
        self,
        sparsity,
        prunen=2,
        prunem=4,
        blocksize=128,
        percdamp=0.01,
    ):
        # assert prunen == 2 and prunem == 4, "VENOM requires 2:4 structured pruning"
        use_venom = (prunen == 2 and prunem == 4)

        if not use_venom:
            self.fasterprune_default(sparsity, prunen, prunem, blocksize,percdamp)
            return

        print("VENOM + Risk-aware Pruning")

        patterns = venom_patterns_2of4(self.dev)
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        if hasattr(self, "quantizer"):
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

        tick = time.time()

        # -------------------------------
        # Finalize Hessian
        # -------------------------------
        H = self.H
        del self.H

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        # -------------------------------
        # Risk-aware damping (Nocedal-style)
        # -------------------------------
        diagH = torch.diag(H)
        damp = percdamp * diagH.mean()
        idx = torch.arange(self.columns, device=self.dev)
        H[idx, idx] += damp

        # -------------------------------
        # Exact inverse Hessian (CUDA-safe)
        # -------------------------------
        L = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(L)
        # Hinv = torch.linalg.cholesky(Hinv, upper=True)

        Losses = torch.zeros(self.rows, device=self.dev)

        patterns = venom_patterns_2of4(self.dev)

        # -------------------------------
        # Blockwise OBS + VENOM
        # -------------------------------
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)

            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(0, count, prunem):
                w = W1[:, i:i + prunem]          # (rows, 4)
                d = Hinv1[i:i + prunem, i:i + prunem]

                # ---- VENOM pattern evaluation (second-order)
                losses = []
                for p in patterns:
                    keep = ~p
                    q = w * keep
                    err = w - q
                    loss = torch.sum(err * err * torch.diag(d), dim=1)
                    losses.append(loss)

                losses = torch.stack(losses, dim=1)  # (rows, 6)
                best = torch.argmin(losses, dim=1)

                q = w.clone()
                for r in range(self.rows):
                    q[r, patterns[best[r]]] = 0

                if hasattr(self, "quantizer"):
                    q = quantize(
                        q.reshape(-1, 1),
                        self.quantizer.scale,
                        self.quantizer.zero,
                        self.quantizer.maxq,
                    ).reshape(w.shape)

                Q1[:, i:i + prunem] = q

                # ---- OBS error propagation (correct)
                Hbb = Hinv1[i:i+prunem, i:i+prunem]      # (4,4)
                Hbb = 0.5 * (Hbb + Hbb.T)

                eigvals = torch.linalg.eigvalsh(Hbb)
                min_eig = eigvals.min()
                if min_eig < 1e-6:
                    Hbb += (1e-6 - min_eig) * torch.eye(prunem, device=Hbb.device)

                # direct block OBS update (single inverse, correct)
                delta = (w - q)

                # propagate ONCE
                W1[:, i:] -= delta @ Hinv1[i:i+prunem, i:]
                Err1[:, i:i+prunem] = delta

                # diagnostic loss (safe, block-consistent)
                Losses += 0.5 * torch.sum((w - q) * (delta @ Hinv1[i:i+prunem, i:i+prunem]))

            W[:, i1:i2] = Q1
            W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

        torch.cuda.synchronize()
        print(f"time {time.time() - tick:.2f}")
        print("error", torch.sum(Losses).item())

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()

        self.layer.weight.data = W.reshape(
            self.layer.weight.shape
        ).to(self.layer.weight.data.dtype)


    def fasterprune_default(
        self, sparsity, prunen=0, prunem=0, blocksize=128, percdamp=.01
    ):
        print("I am in wrong path")
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        if hasattr(self, 'quantizer'):
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        mask = None

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prunen == 0: 
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prunen != 0 and i % prunem == 0:
                    tmp = W1[:, i:(i + prunem)] ** 2 / (torch.diag(Hinv1)[i:(i + prunem)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prunen, dim=1, largest=False)[1], True)

                q = w.clone()
                q[mask1[:, i]] = 0

                if hasattr(self, 'quantizer'):
                    q = quantize(
                        q.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                    ).flatten()

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = W[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()
