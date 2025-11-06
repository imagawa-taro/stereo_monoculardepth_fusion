import numpy as np
from typing import Tuple, List, Optional
import time


# ====== 2D ヘルパ ======

def make_weights_2d(da: np.ndarray, threshold: float = 1.0, eps_small: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    """
    ガイド画像daの勾配に応じて異方性重みを作る。
    - 横方向(wx): daのx差分に基づき、強いエッジでは小さく
    - 縦方向(wy): daのy差分に基づき、強いエッジでは小さく
    """
    da = np.asarray(da, dtype=np.float32)
    H, W = da.shape
    # 前進差分の絶対値（エッジ強度）
    gx = np.zeros((H, W-1), dtype=np.float32)
    gy = np.zeros((H-1, W), dtype=np.float32)
    gx[:] = np.abs(da[:, 1:] - da[:, :-1])
    gy[:] = np.abs(da[1:, :] - da[:-1, :])

    # ピクセルに対応するwx, wyを作る
    wx = np.ones((H, W), dtype=np.float32)
    wy = np.ones((H, W), dtype=np.float32)

    # 横方向は左ピクセルの重みでエッジを評価（右端列はフラックス0なので未使用）
    wx[:, :-1][gx > threshold] = eps_small
    wx[:, -1] = 0.0  # Neumannで右端の外側フラックスは常に0

    # 縦方向は上ピクセルの重みでエッジを評価（下端行はフラックス0なので未使用）
    wy[:-1, :][gy > threshold] = eps_small
    wy[-1, :] = 0.0  # Neumannで下端の外側フラックスは常に0

    return wx, wy

def compute_b_2d(J1: np.ndarray, J2: np.ndarray, wx: np.ndarray, wy: np.ndarray) -> np.ndarray:
    """
    b = J1 - div( wx * ∂x J2 + wy * ∂y J2 )（Neumann境界：外側フラックス0）
    ∂x,∂yは前進差分、divは後退差分（qの外側は0パディング）
    """
    J1 = np.asarray(J1, dtype=np.float32)
    J2 = np.asarray(J2, dtype=np.float32)
    wx = np.asarray(wx, dtype=np.float32)
    wy = np.asarray(wy, dtype=np.float32)
    H, W = J1.shape
    # qx: H x (W-1), qy: (H-1) x W
    diffx = J2[:, 1:] - J2[:, :-1]
    diffy = J2[1:, :] - J2[:-1, :]
    qx = wx[:, :-1] * diffx
    qy = wy[:-1, :] * diffy

    # div_x（左右に0を付加して後退差分）
    qxpad = np.zeros((H, W+1), dtype=np.float32)
    qxpad[:, 1:W] = qx
    div_x = qxpad[:, 1:] - qxpad[:, :-1]  # H x W

    # div_y（上下に0を付加して後退差分）
    qypad = np.zeros((H+1, W), dtype=np.float32)
    qypad[1:H, :] = qy
    div_y = qypad[1:, :] - qypad[:-1, :]  # H x W

    return J1 - (div_x + div_y)

def apply_A_2d(x: np.ndarray, wx: np.ndarray, wy: np.ndarray) -> np.ndarray:
    """
    A x = x - div( wx * ∂x x + wy * ∂y x )
    Neumann境界：外側フラックス0
    """
    x = np.asarray(x, dtype=np.float32)
    wx = np.asarray(wx, dtype=np.float32)
    wy = np.asarray(wy, dtype=np.float32)
    H, W = x.shape
    diffx = x[:, 1:] - x[:, :-1]           # H x (W-1)
    diffy = x[1:, :] - x[:-1, :]           # (H-1) x W
    qx = wx[:, :-1] * diffx
    qy = wy[:-1, :] * diffy
    qxpad = np.zeros((H, W+1), dtype=np.float32)
    qxpad[:, 1:W] = qx
    div_x = qxpad[:, 1:] - qxpad[:, :-1]
    qypad = np.zeros((H+1, W), dtype=np.float32)
    qypad[1:H, :] = qy
    div_y = qypad[1:, :] - qypad[:-1, :]
    return x - (div_x + div_y)

def jacobi_smooth_2d(
    x: np.ndarray,
    b: np.ndarray,
    wx: np.ndarray,
    wy: np.ndarray,
    iters: int = 1,
    omega: float = 0.9
) -> np.ndarray:
    """
    重み付きJacobiスムーザー（Neumann）
    2Dの5点ステンシル：
      d = 1 + wR + wL + wD + wU
      x_new = ( b + wRxR + wLxL + wDxD + wUxU ) / d
    """
    x = x.astype(np.float32, copy=True)
    b = b.astype(np.float32, copy=False)
    wx = wx.astype(np.float32, copy=False)
    wy = wy.astype(np.float32, copy=False)
    H, W = x.shape
    # 係数（境界で外側フラックス0）
    wR = wx.copy(); wR[:, -1] = 0.0
    wL = np.zeros_like(wx); wL[:, 1:] = wx[:, :-1]  # 左
    wD = wy.copy(); wD[-1, :] = 0.0                # 下
    wU = np.zeros_like(wy); wU[1:, :] = wy[:-1, :] # 上
    d = 1.0 + wR + wL + wD + wU

    for _ in range(iters):
        # 近傍値（境界は複製、ただし係数側で0が掛かるため安全）
        xR = np.empty_like(x); xR[:, :-1] = x[:, 1:]; xR[:, -1] = x[:, -1]
        xL = np.empty_like(x); xL[:, 1:]  = x[:, :-1]; xL[:, 0]  = x[:, 0]
        xD = np.empty_like(x); xD[:-1, :] = x[1:, :];  xD[-1, :] = x[-1, :]
        xU = np.empty_like(x); xU[1:, :]  = x[:-1, :]; xU[0, :]  = x[0, :]

        x_new = (b + wR * xR + wL * xL + wD * xD + wU * xU) / d
        x = (1.0 - omega) * x + omega * x_new
    return x

# ====== 2D マルチグリッド ======
def smooth_downsample_2d(arr: np.ndarray) -> np.ndarray:
    """
    2Dの3x3平滑（[1,2,1;2,4,2;1,2,1]/16）＋2倍間引き（端は複製）
    """
    arr = np.asarray(arr, dtype=np.float32)
    H, W = arr.shape
    if H == 0 or W == 0:
        return arr.copy()
    # 行方向平滑（複製パディング）
    a = np.empty((H + 2, W), dtype=np.float32)
    a[0, :] = arr[0, :]
    a[1:H+1, :] = arr
    a[H+1, :] = arr[-1, :]
    tmp = 0.25 * a[:-2, :] + 0.5 * a[1:-1, :] + 0.25 * a[2:, :]

    # 列方向平滑（複製パディング）
    b = np.empty((H, W + 2), dtype=np.float32)
    b[:, 0] = tmp[:, 0]
    b[:, 1:W+1] = tmp
    b[:, W+1] = tmp[:, -1]
    sm = 0.25 * b[:, :-2] + 0.5 * b[:, 1:-1] + 0.25 * b[:, 2:]
    return sm[::2, ::2].copy()

def prolong_bilinear_2d(coarse: np.ndarray, fine_shape: Tuple[int, int]) -> np.ndarray:
    """
    2Dのバイリニア補間延長。coarse(Hc,Wc) → fine(Hf,Wf)（おおむね2倍）
    """
    coarse = np.asarray(coarse, dtype=np.float32)
    Hc, Wc = coarse.shape
    Hf, Wf = fine_shape
    if Hc == 0 or Wc == 0:
        return np.zeros((Hf, Wf), dtype=np.float32)
    if Hc == 1 and Wc == 1:
        return np.full((Hf, Wf), coarse[0, 0], dtype=np.float32)

    Ht, Wt = 2 * Hc - 1, 2 * Wc - 1
    fine_est = np.empty((Ht, Wt), dtype=np.float32)

    # 既知点
    fine_est[0::2, 0::2] = coarse
    # 垂直中央
    if Hc > 1:
        fine_est[1::2, 0::2] = 0.5 * (coarse[:-1, :] + coarse[1:, :])
    # 水平中央
    if Wc > 1:
        fine_est[0::2, 1::2] = 0.5 * (coarse[:, :-1] + coarse[:, 1:])
    # センタ（四点平均）
    if Hc > 1 and Wc > 1:
        fine_est[1::2, 1::2] = 0.25 * (coarse[:-1, :-1] + coarse[1:, :-1] + coarse[:-1, 1:] + coarse[1:, 1:])

    # 必要サイズへ調整（不足分は端の値で埋める）
    out = fine_est
    if out.shape[0] < Hf:
        pad_rows = np.tile(out[-1:, :], (Hf - out.shape[0], 1))
        out = np.concatenate([out, pad_rows], axis=0)
    if out.shape[1] < Wf:
        pad_cols = np.tile(out[:, -1:], (1, Wf - out.shape[1]))
        out = np.concatenate([out, pad_cols], axis=1)
    return out[:Hf, :Wf].copy()

def build_weight_pyramid_2d(
    wx0: np.ndarray,
    wy0: np.ndarray,
    max_levels: Optional[int] = None
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    wx/wyピラミッド（各レベルで 1/4 スケール）を作る
    """
    wx_levels = [wx0.astype(np.float32, copy=True)]
    wy_levels = [wy0.astype(np.float32, copy=True)]
    L = 1
    while True:
        if max_levels is not None and L >= max_levels:
            break
        prevx = wx_levels[-1]; prevy = wy_levels[-1]
        H, W = prevx.shape
        if H <= 4 or W <= 4:
            break
        nxtx = smooth_downsample_2d(prevx) * 0.25
        nxty = smooth_downsample_2d(prevy) * 0.25
        wx_levels.append(nxtx)
        wy_levels.append(nxty)
        L += 1
    return wx_levels, wy_levels

def v_cycle_2d(
    x: np.ndarray,
    b: np.ndarray,
    wx_levels: List[np.ndarray],
    wy_levels: List[np.ndarray],
    level: int = 0,
    pre: int = 2,
    post: int = 2,
    omega: float = 0.9,
    coarse_iters: int = 40
) -> np.ndarray:
    """
    2D Vサイクル（誤差方程式の標準実装）
    - x: 現レベルの解画像
    - b: 現レベルの右辺
    - wx_levels[level], wy_levels[level]: 現レベルの重み
    """
    wx = wx_levels[level]
    wy = wy_levels[level]
    H, W = wx.shape
    # プレスムーズ
    x = jacobi_smooth_2d(x, b, wx, wy, iters=pre, omega=omega)
    # 残差
    r = b - apply_A_2d(x, wx, wy)

    # 収束 or 最粗レベル条件
    is_coarsest = (level == len(wx_levels) - 1) or (min(H, W) <= 8)
    if is_coarsest:
        e = np.zeros_like(x)
        e = jacobi_smooth_2d(e, r, wx, wy, iters=coarse_iters, omega=omega)
        x = x + e
        x = jacobi_smooth_2d(x, b, wx, wy, iters=max(1, post//2), omega=omega)
        return x

    # 制限（残差）
    r_coarse = smooth_downsample_2d(r)
    # 粗レベルで誤差方程式 A e = r を解く
    x0 = np.zeros_like(r_coarse)
    e_coarse = v_cycle_2d(x0, r_coarse, wx_levels, wy_levels, level+1, pre=pre, post=post, omega=omega, coarse_iters=coarse_iters)
    # 延長して修正
    e_fine = prolong_bilinear_2d(e_coarse, (H, W))
    x = x + e_fine
    # ポストスムーズ
    x = jacobi_smooth_2d(x, b, wx, wy, iters=post, omega=omega)
    return x

def restore_2d_mg(
    pd: np.ndarray,
    da: np.ndarray,
    lam: float = 100.0,
    threshold: float = 1.0,
    levels: Optional[int] = None,
    cycles: int = 2,
    pre: int = 2,
    post: int = 2,
    omega: float = 0.9,
    coarse_iters: int = 40,
    clip_range: Optional[Tuple[float, float]] = None,
    verbose: bool = True
) -> np.ndarray:
    """
    2Dマルチグリッド解法（Vサイクル×cycles）
    - A x = b, A = I - div(wx ∂x· + wy ∂y·), wx = lam * Wx, wy = lam * Wy
    - wx/wyは等方なら同じW2から作ってよい
    """
    pd = np.asarray(pd, dtype=np.float32)
    da = np.asarray(da, dtype=np.float32)
    H, W = pd.shape
    assert da.shape == pd.shape

    # 重み（異方性）。閾値で強エッジを抑制
    wx_base, wy_base = make_weights_2d(da, threshold=threshold, eps_small=1e-3)
    wx0 = lam * np.clip(wx_base, 0.0, np.finfo(np.float32).max)
    wy0 = lam * np.clip(wy_base, 0.0, np.finfo(np.float32).max)

    # クリップ範囲
    if clip_range is None:
        vmin = float(min(pd.min(), da.min()))
        vmax = float(max(pd.max(), da.max()))
    else:
        vmin, vmax = clip_range

    # ピラミッド（wx/wy）
    wx_levels, wy_levels = build_weight_pyramid_2d(wx0, wy0, max_levels=levels)

    # 右辺（最細で明示計算）
    b = compute_b_2d(pd, da, wx0, wy0)

    # 初期解（pdを採用）
    x = pd.copy()

    t0 = time.time()
    for c in range(cycles):
        x = v_cycle_2d(x, b, wx_levels, wy_levels, level=0, pre=pre, post=post, omega=omega, coarse_iters=coarse_iters)
        x = np.clip(x, vmin, vmax)
        if verbose:
            res = b - apply_A_2d(x, wx0, wy0)
            rn = np.linalg.norm(res) / (np.sqrt(H*W) + 1e-12)
            print(f"V-cycle {c+1}/{cycles}: residual RMS = {rn:.6e}")
    if verbose:
        print(f"MG elapsed: {time.time() - t0:.3f} sec")
    return x


# ====== デモ用（簡易トイ画像） ======
def _toy_image(H=512, W=512, seed=0):
    rng = np.random.RandomState(seed)
    gt = np.zeros((H, W), dtype=np.float32)
    gt[H//8:3*H//8, W//8:3*W//8] = 1.0
    gt[H//2:7*H//8, W//4:W//2] = 0.5
    # ガイド（エッジ強調＋ノイズ）
    da = gt * 1.02 + 0.03 * rng.randn(H, W).astype(np.float32)
    # 観測（少しバイアス＋ノイズ）
    pd = gt * 0.95 + 0.05 * rng.randn(H, W).astype(np.float32)
    return gt, pd, da


# ====== メインデモ ======
if __name__ == "__main__":
    try:
        from set_data_2d import set_gt_img, set_pd_img, set_da_img
        H, W = 1026, 2048
        gt = set_gt_img(H, W); da = set_da_img(gt); pd = set_pd_img(gt)
    except Exception:
        print("set_data_2d.py not found, using toy data.")
        gt, pd, da = _toy_image(2048, 4096)


    # lamの目安：強めに正則化すると伝搬が安定（画像なら ~max(H,W)〜5*max(H,W) を試す）
    x_mg = restore_2d_mg(pd, da, lam=2000.0, threshold=0.2,
                        levels=None, cycles=3, pre=2, post=2, omega=0.9, coarse_iters=50,
                        clip_range=None, verbose=True)

    mse_pd = np.mean((pd - gt)**2)
    mse_da = np.mean((da - gt)**2)
    mse_mg = np.mean((x_mg - gt)**2)
    print(f"MSE pd={mse_pd:.6f}, da={mse_da:.6f}, MG={mse_mg:.6f}")

    # 可視化（必要なら）
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 10))
        plt.subplot(2, 2, 1); plt.title('Ground Truth'); plt.imshow(gt, cmap='gray'); plt.axis('off')
        plt.subplot(2, 2, 2); plt.title('Observed pd');   plt.imshow(pd, cmap='gray'); plt.axis('off')
        plt.subplot(2, 2, 3); plt.title('Monocular da');  plt.imshow(da, cmap='gray'); plt.axis('off')
        plt.subplot(2, 2, 4); plt.title('Restored MG');   plt.imshow(x_mg, cmap='gray'); plt.axis('off')
        plt.tight_layout(); plt.show()
    except Exception as e:
        print("Matplotlib not available:", e)
