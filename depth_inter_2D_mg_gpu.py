import numpy as np
from typing import Tuple, List, Optional
import time

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


# ====== 2D 重み作成 ======

def make_weights_2d(da: np.ndarray, threshold: float = 1.0, eps_small: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    """
    ガイド画像daの勾配に応じて異方性重みを作成。
    - 水平方向(wx): daのx方向に勾配が大きいと重みが小さくなる
    - 垂直方向(wy): daのy方向に勾配が大きいと重みが小さくなる
    """
    da = np.asarray(da, dtype=np.float32)
    H, W = da.shape
    # 勾配強度の計算（エッジ強さ）
    gx = np.zeros((H, W-1), dtype=np.float32)
    gy = np.zeros((H-1, W), dtype=np.float32)
    gx[:] = np.abs(da[:, 1:] - da[:, :-1])
    gy[:] = np.abs(da[1:, :] - da[:-1, :])

    # 初期重みとしてwx, wyを1で初期化
    wx = np.ones((H, W), dtype=np.float32)
    wy = np.ones((H, W), dtype=np.float32)

    # 水平方向は右隣の重みを調整（右端はNeumann境界なので重み0）
    wx[:, :-1][gx > threshold] = eps_small
    wx[:, -1] = 0.0  # Neumannで右端の境界フラックスは常に0

    # 垂直方向は下隣の重みを調整（下端はNeumann境界なので重み0）
    wy[:-1, :][gy > threshold] = eps_small
    wy[-1, :] = 0.0  # Neumannで下端の境界フラックスは常に0

    return wx, wy

def compute_b_2d(J1: np.ndarray, J2: np.ndarray, wx: np.ndarray, wy: np.ndarray) -> np.ndarray:
    """
    b = J1 - div( wx * ∇x J2 + wy * ∇y J2 )（Neumann境界でフラックス0）
    ∇x,∇yは差分、divは発散（qの境界は0でパディング）
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

    # div_x（左右端は0を拡張して発散を計算）
    qxpad = np.zeros((H, W+1), dtype=np.float32)
    qxpad[:, 1:W] = qx
    div_x = qxpad[:, 1:] - qxpad[:, :-1]  # H x W

    # div_y（上下端は0を拡張して発散を計算）
    qypad = np.zeros((H+1, W), dtype=np.float32)
    qypad[1:H, :] = qy
    div_y = qypad[1:, :] - qypad[:-1, :]  # H x W

    return J1 - (div_x + div_y)

def apply_A_2d(x: np.ndarray, wx: np.ndarray, wy: np.ndarray) -> np.ndarray:
    """
    A x = x - div( wx * ∇x x + wy * ∇y x )
    Neumann境界でフラックス0
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
    異方性Jacobiスムーザー（Neumann）
    2Dの5点ステンシル：
      d = 1 + wR + wL + wD + wU
      x_new = ( b + wR xR + wL xL + wD xD + wU xU ) / d
    """
    x = x.astype(np.float32, copy=True)
    b = b.astype(np.float32, copy=False)
    wx = wx.astype(np.float32, copy=False)
    wy = wy.astype(np.float32, copy=False)
    H, W = x.shape
    # 隣接重み（境界はフラックス0）
    wR = wx.copy(); wR[:, -1] = 0.0
    wL = np.zeros_like(wx); wL[:, 1:] = wx[:, :-1]  # 左
    wD = wy.copy(); wD[-1, :] = 0.0                # 下
    wU = np.zeros_like(wy); wU[1:, :] = wy[:-1, :] # 上
    d = 1.0 + wR + wL + wD + wU

    for _ in range(iters):
        # 隣接値（境界は反射、つまり外側は0でなく端値を保持）
        xR = np.empty_like(x); xR[:, :-1] = x[:, 1:]; xR[:, -1] = x[:, -1]
        xL = np.empty_like(x); xL[:, 1:]  = x[:, :-1]; xL[:, 0]  = x[:, 0]
        xD = np.empty_like(x); xD[:-1, :] = x[1:, :];  xD[-1, :] = x[-1, :]
        xU = np.empty_like(x); xU[1:, :]  = x[:-1, :]; xU[0, :]  = x[0, :]

        x_new = (b + wR * xR + wL * xL + wD * xD + wU * xU) / d
        x = (1.0 - omega) * x + omega * x_new
    return x


# ====== Torch 実装 ======

def torch_apply_A_2d(x: torch.Tensor, wx: torch.Tensor, wy: torch.Tensor) -> torch.Tensor:
    """
    Torch 実装: A x = x - div( wx ∇x + wy ∇y )
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is not available; install PyTorch to use torch_apply_A_2d.")
    x = x.to(torch.float32)
    wx = wx.to(torch.float32)
    wy = wy.to(torch.float32)
    H, W = x.shape

    diffx = x[:, 1:] - x[:, :-1]
    diffy = x[1:, :] - x[:-1, :]
    qx = wx[:, :-1] * diffx
    qy = wy[:-1, :] * diffy

    qxpad = torch.zeros((H, W + 1), dtype=x.dtype, device=x.device)
    qxpad[:, 1:W] = qx
    div_x = qxpad[:, 1:] - qxpad[:, :-1]

    qypad = torch.zeros((H + 1, W), dtype=x.dtype, device=x.device)
    qypad[1:H, :] = qy
    div_y = qypad[1:, :] - qypad[:-1, :]

    return x - (div_x + div_y)


def torch_jacobi_smooth_2d(
    x: torch.Tensor,
    b: torch.Tensor,
    wx: torch.Tensor,
    wy: torch.Tensor,
    iters: int = 1,
    omega: float = 0.9,
) -> torch.Tensor:
    """
    Torch 実装のJacobiスムーザー。
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is not available; install PyTorch to use torch_jacobi_smooth_2d.")
    x = x.to(torch.float32).clone()
    b = b.to(torch.float32)
    wx = wx.to(torch.float32)
    wy = wy.to(torch.float32)

    wR = wx.clone()
    wR[:, -1] = 0.0
    wL = torch.zeros_like(wx)
    wL[:, 1:] = wx[:, :-1]

    wD = wy.clone()
    wD[-1, :] = 0.0
    wU = torch.zeros_like(wy)
    wU[1:, :] = wy[:-1, :]

    d = 1.0 + wR + wL + wD + wU

    for _ in range(iters):
        xR = torch.empty_like(x)
        xR[:, :-1] = x[:, 1:]
        xR[:, -1] = x[:, -1]

        xL = torch.empty_like(x)
        xL[:, 1:] = x[:, :-1]
        xL[:, 0] = x[:, 0]

        xD = torch.empty_like(x)
        xD[:-1, :] = x[1:, :]
        xD[-1, :] = x[-1, :]

        xU = torch.empty_like(x)
        xU[1:, :] = x[:-1, :]
        xU[0, :] = x[0, :]

        x_new = (b + wR * xR + wL * xL + wD * xD + wU * xU) / d
        x = (1.0 - omega) * x + omega * x_new

    return x


# ====== 2D マルチグリッド ======
# ... （以下、既存の smooth_downsample_2d や prolong_bilinear_2d、
#      build_weight_pyramid_2d、v_cycle_2d、restore_2d_mg などの実装は元ファイルのまま続きます） ...


def run_torch_demo(device: str) -> None:
    """
    NumPy 実装との一致と速度を簡易チェック。
    """
    print(f"\n--- device: {device} ---")
    torch_device = torch.device(device)

    H, W = 4000, 6000
    from depth_inter_2D_mg import _toy_image  # 元の NumPy 関数を再利用
    gt, pd, da = _toy_image(H, W, seed=0)

    lam = 2000.0
    wx_base, wy_base = make_weights_2d(da, threshold=0.2, eps_small=1e-3)
    wx_np = lam * np.clip(wx_base, 0.0, np.finfo(np.float32).max)
    wy_np = lam * np.clip(wy_base, 0.0, np.finfo(np.float32).max)
    b_np = compute_b_2d(pd, da, wx_np, wy_np)
    x_np = pd.copy()

    wx = torch.from_numpy(wx_np).to(torch_device)
    wy = torch.from_numpy(wy_np).to(torch_device)
    b = torch.from_numpy(b_np).to(torch_device)
    x = torch.from_numpy(x_np).to(torch_device)

    with torch.no_grad():
        x_torch = torch_jacobi_smooth_2d(x, b, wx, wy, iters=5, omega=0.9)
        Ax_torch = torch_apply_A_2d(x_torch, wx, wy)

    x_ref_np = jacobi_smooth_2d(x_np, b_np, wx_np, wy_np, iters=5, omega=0.9)
    Ax_ref_np = apply_A_2d(x_ref_np, wx_np, wy_np)

    x_err = torch.max(torch.abs(x_torch.cpu() - torch.from_numpy(x_ref_np)))
    Ax_err = torch.max(torch.abs(Ax_torch.cpu() - torch.from_numpy(Ax_ref_np)))
    print(f"max |x_torch - x_numpy| = {x_err.item():.3e}")
    print(f"max |Ax_torch - Ax_numpy| = {Ax_err.item():.3e}")

    timing_runs = 10
    with torch.no_grad():
        if torch_device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        tmp = x.clone()
        for _ in range(timing_runs):
            tmp = torch_jacobi_smooth_2d(tmp, b, wx, wy, iters=5, omega=0.9)
        if torch_device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    print(f"{timing_runs} Jacobi iterations: {elapsed:.3f} s")


if __name__ == "__main__":
    # README: NumPy/torch両方で一致性を確認しつつ簡易ベンチマークを走らせる例
    if TORCH_AVAILABLE:
        run_torch_demo("cpu")
        if torch.cuda.is_available():
            run_torch_demo("cuda")
        else:
            print("\nCUDA 未対応環境のため GPU 検証をスキップしました。")
    else:
        print("torch がインストールされていないため、GPU 検証をスキップしました。")
