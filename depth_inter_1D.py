import numpy as np
import time
import matplotlib.pyplot as plt

def make_weights_1d(da, threshold=1.0):
    """
    1次元の位置重みを作る。
    - W1: データ項の重み（全て1）
    - W2: 勾配項の重み。da の差分が大きい箇所では小さくする。
      2次元版のW2[..., :, 1:]に合わせ、ペアの右側位置に重みが乗るイメージでW2[1:]を使用。
    """
    N = len(da)
    W1 = np.ones(N, dtype=np.float32)


    W2 = np.ones(N, dtype=np.float32)
    da_dx = np.abs(np.diff(da, prepend=da[:1]))  # 長さN、先頭は0差分
    # 差分が大きい場所の右側インデックスに対応する重みを小さく
    W2[da_dx > threshold] = 0.01

    return W1, W2

def compute_cost_and_grad_1d(x, pd, da, W1, W2, lam):
    """
    1次元版のコストと勾配を計算。
    C1 = mean(W1 * (x - pd)^2)
    C2 = mean(W2[1:] * ((x[1:] - x[:-1]) - (da[1:] - da[:-1]))^2)
    C = C1 + lam * C2


    勾配は解析的に導出して実装。
    """
    N = len(x)
    assert len(pd) == N and len(da) == N and len(W1) == N and len(W2) == N

    # C1
    diff_data = x - pd
    C1 = np.mean(W1 * diff_data**2)

    # C2
    diff_x  = x[1:]  - x[:-1]
    diff_da = da[1:] - da[:-1]
    err = diff_x - diff_da               # 長さ N-1
    W2p = W2[1:]                          # ペア重み（右側に合わせてクロップ）
    C2 = np.mean(W2p * err**2)

    # 勾配
    grad = np.zeros_like(x, dtype=np.float32)

    # dC1/dx
    grad += (2.0 / N) * W1 * diff_data

    # dC2/dx
    # C2 = mean(W2p * err^2) = (1/(N-1)) * sum(W2p * err^2)
    # 各 x_i に対して、err[p] = (x[p+1]-x[p]) - (da[p+1]-da[p])
    # d err[p] / d x[p]   = -1
    # d err[p] / d x[p+1] = +1
    t = W2p * err  # 長さ N-1
    grad_c2 = np.zeros_like(x, dtype=np.float32)
    if N > 1:
        # i = 0
        grad_c2[0] += -t[0]
        # 中間 i = 1..N-2
        if N > 2:
            grad_c2[1:-1] += t[:-1] - t[1:]
        # i = N-1
        grad_c2[-1] += t[-1]

    grad += lam * (2.0 / (N - 1)) * grad_c2 if N > 1 else 0.0

    C = C1 + lam * C2
    return C1, C2, C, grad

def restore_1d(pd, da, lam=100.0, lr=0.1, num_iters=1000, threshold=1.0, clip_range=None, verbose=True):
    """
    1次元信号の復元。
    - pd: 観測（ステレオ視差に相当）
    - da: ガイド（単眼デプスに相当、勾配構造を保つ）
    - lam: 勾配項の重み
    - lr: 学習率（勾配降下法）
    - num_iters: 反復回数
    - threshold: W2作成時の差分閾値
    - clip_range: (min, max) で復元値をクリップ。None の場合は pd と da のレンジを統合して使用。

    戻り値:
    - x_rest: 復元信号
    - loss_history: 各反復のコスト履歴（リスト）
    """
    pd = np.asarray(pd, dtype=np.float32).copy()
    da = np.asarray(da, dtype=np.float32).copy()
    N = len(pd)
    assert len(da) == N


    # 位置重み
    W1, W2 = make_weights_1d(da, threshold=threshold)

    # 初期値は da（元コードに合わせる）
    x = da.copy()

    # クリップ範囲
    if clip_range is None:
        vmin = float(min(pd.min(), da.min()))
        vmax = float(max(pd.max(), da.max()))
    else:
        vmin, vmax = clip_range

    loss_history = []
    # 反復最適化ループ
    t0 = time.time()
    for it in range(1, num_iters + 1):
        C1, C2, C, grad = compute_cost_and_grad_1d(x, pd, da, W1, W2, lam)
        loss_history.append(C)

        # 勾配降下
        x -= lr * grad

        # クリップ
        x = np.clip(x, vmin, vmax)

        # ログ
        if verbose and (it == 1 or it % 50 == 0 or it == num_iters):
            print(f"iter {it:4d} | C={C:.6f} | C1={C1:.6f} | C2={C2:.6f}")

    if verbose:
        print(f"Elapsed time: {time.time() - t0:.3f} sec")
    return x, loss_history


def set_gt(N):
    """
    デモ用のGT信号を設定
    gt: 段差を含む信号
    """
    ww = N // 5
    gt = np.zeros(N, dtype=np.float32)
    for i in range(ww):
        gt[ww + i] = 30 + 15.0 * np.sin(np.pi * i / ww)  # 半円形
        gt[3*ww + i] = 30 + 15.0 * np.sin(np.pi * i / ww)  # 半円形
    return gt


def set_pd(gt):
    """
    デモ用の観測信号 pd を設定
    pd: gt をぼかし＋量子化＋ノイズ
    """
    np.random.seed(1)
    # ぼかし
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size
    pd = gt.copy()
    pd = np.convolve(pd, kernel, mode='same')
    # 量子化
    pd = np.round(pd / 5.0) * 5.0
    # ノイズ
    pd += np.random.normal(0, 2.0, size=len(pd)).astype(np.float32)
    return pd

def set_da(gt):
    """
    デモ用のガイド信号 da を設定
    da: gt の後ろ半分の段差を10低くする
    """
    da = gt.copy()
    N = len(da)
    da[N//2:] = np.maximum(da[N//2:] - 20, 0)
    return da

def demo():
    """
    簡単なデモ用。
    da: 形状ガイド（段差を含む信号）
    pd: 観測（daとスケールがズレておりノイズも乗っている）
    """
    N = 1000
    # GTの設定（デモ用にdaをGTとする）
    gt = set_gt(N)
    # DAの設定
    da = set_da(gt)
    # PDの設定
    pd = set_pd(gt)

    # 復元
    x_rest, loss_history = restore_1d(pd, da, lam=100.0, lr=1, num_iters=800, threshold=8.0, verbose=True)

    # プロット表示
    plt.figure(figsize=(10, 6))
    plt.plot(gt, label='Ground Truth (gt)', linestyle='-', marker='o', markersize=2)
    plt.plot(pd, label='Stereo (pd)', linestyle='--', marker='o', markersize=2)
    plt.plot(da, label='Monocular depth (da)', linestyle='-.', marker='s', markersize=2)
    plt.plot(x_rest, label='Restored', linestyle='-', marker='x', markersize=2)
    plt.legend()
    plt.title('1D Depth Restoration Demo')
    plt.xlabel('Sample Index')
    plt.ylabel('Depth Value')
    plt.grid()

    plt.figure(figsize=(6,4))
    plt.plot(loss_history)
    plt.title('Loss History')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    demo()
