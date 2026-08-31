#!/usr/bin/env python3
"""N₂ 6-31G frozen-core wavepacket analysis.
Uses the full CI vector from direct_spin1 (alpha-major, dim = C(16,5)^2 = 1.9e7)
and decomposes it by excitation order relative to the RHF reference,
plus the amplitude-concentration curve (the quantitative "wavepacket" picture:
how many states carry 99.9% of the norm at each geometry)."""
import sys
import numpy as np
from pyscf.fci import cistring

def main():
    npy = sys.argv[1]       # e.g. _n2_c631g_14.npy
    n_act = int(sys.argv[2])  # 16
    na = int(sys.argv[3])     # 5
    strs = np.asarray(cistring.gen_strings4orblist(range(n_act), na), dtype=np.int64)
    nb = len(strs)
    c = np.load(npy)
    if c.ndim == 2:
        assert c.shape == (nb, nb), f'expected {(nb, nb)}, got {c.shape}'
    else:
        assert len(c) == nb * nb, f'expected {nb*nb}, got {len(c)}'
        c = c.reshape(nb, nb)
    C = c.reshape(nb, nb)
    W = np.abs(C) ** 2
    tot = W.sum()
    hf = (1 << na) - 1
    ia_hf = int(np.where(strs == hf)[0][0])
    pc = np.array([bin(int(x)).count('1') for x in strs ^ hf])
    K = (pc[:, None] + pc[None, :]) // 2
    print(f'dim={nb*nb}  HF|c|^2 = {W[ia_hf, ia_hf]/tot:.6f} ({100*W[ia_hf,ia_hf]/tot:.3f}%)')
    for k in range(0, 10):
        w = W[K == k].sum()
        if w / tot > 1e-7:
            print(f'  k={k}: {w:.6f} ({100*w/tot:.4f}%)')
    flat = np.sort(W.ravel())[::-1]
    cum = np.cumsum(flat)
    print('concentration:')
    for frac in (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1):
        kk = max(int(frac * len(flat)), 1)
        print(f'  top {100*frac:.4f}% states ({kk}) hold {100*cum[kk-1]/tot:.6f}% norm')
    # effective dimension: states for 99.9% and 99.99%
    for target in (0.999, 0.9999, 0.99999):
        kk = int(np.searchsorted(cum, target * tot)) + 1
        print(f'  eff-dim for {100*target:.3f}% norm: {kk} ({100*kk/len(flat):.4f}% of dim)')

if __name__ == '__main__':
    main()
