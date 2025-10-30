import numpy as np
import time

# ====== 共通ヘルパ ======

def make_weights_1d(da, threshold=1.0):
    N = len(da)
    W1 = np.ones(N, dtype=np.float32)
    W2 = np.ones(N, dtype=np.float32)
    da_dx = np.abs(np.diff(da, prepend=da[:1]))
    W2[da_dx > threshold] = 0.001
    return W1, W2

def compute_b_1d(J1, J2, wx):
    """
    b = J1 - div( wx * ∂x J2 ) （Neumann境界：qの外側は0）
    ∂xは前進差分、divは後退差分
    """
    J1 = np.asarray(J1, dtype=np.float32)
    J2 = np.asarray(J2, dtype=np.float32)
    wx = np.asarray(wx, dtype=np.float32)
    N = len(J1)
    if N == 0:
        return J1.copy()
    # q[i] は i=0..N-2 で定義（最後は使わない）
    diffJ2 = J2[1:] - J2[:-1]
    q = wx[:-1] * diffJ2  # 長さ N-1
    # パディングして div = qpad[i] - qpad[i-1]
    qpad = np.zeros(N + 1, dtype=np.float32)
    qpad[1:N] = q
    div = qpad[1:] - qpad[:-1]  # 長さ N
    b = J1 - div
    return b

def apply_A_1d(x, wx):
    """
    A x = x - div( wx * ∂x x )
    Neumann境界：外側フラックス0
    """
    x = np.asarray(x, dtype=np.float32)
    wx = np.asarray(wx, dtype=np.float32)
    N = len(x)
    if N == 0:
        return x.copy()
    diffx = x[1:] - x[:-1]           # 長さ N-1
    q = wx[:-1] * diffx              # 長さ N-1
    qpad = np.zeros(N + 1, dtype=np.float32)
    qpad[1:N] = q
    div = qpad[1:] - qpad[:-1]       # 長さ N
    return x - div

def jacobi_smooth_1d(x, b, wx, iters=1, omega=0.9):
    """
    重み付きJacobiスムーザー（Neumann）
    x^{new} = (1-ω) x + ω * ( b + wE*x[i+1] + wW*x[i-1] ) / d
    d = 1 + wE + wW, wE[N-1]=0, wW[0]=0
    """
    x = x.astype(np.float32, copy=True)
    b = b.astype(np.float32, copy=False)
    wx = wx.astype(np.float32, copy=False)
    N = len(x)
    if N == 0:
        return x
    # 係数を前計算
    wE = wx.copy()
    wW = np.empty_like(wx)
    wW[0] = 0.0
    wW[1:] = wx[:-1]
    wE[-1] = 0.0
    d = 1.0 + wE + wW

    for _ in range(iters):
        xL = np.empty_like(x); xR = np.empty_like(x)
        xL[0] = x[0];      xL[1:] = x[:-1]
        xR[:-1] = x[1:];   xR[-1] = x[-1]
        x_new = (b + wE * xR + wW * xL) / d
        x = (1.0 - omega) * x + omega * x_new
    return x

# ====== 1D マルチグリッド ======

def smooth_downsample_1d(arr):
    """
    1Dの[0.25, 0.5, 0.25]平滑＋2倍間引き（端は反射/複製）
    """
    if len(arr) <= 1:
        return arr.copy()
    arr = np.asarray(arr, dtype=np.float32)
    padL = arr[0:1]
    padR = arr[-1:]
    a = np.concatenate([padL, arr, padR])
    sm = 0.25 * a[:-2] + 0.5 * a[1:-1] + 0.25 * a[2:]
    return sm[::2].copy()

def prolong_linear_1d(coarse, fine_len):
    """
    1Dの線形補間延長。coarse長Nc→fine長 fine_len （おおむね 2*Nc に一致）
    """
    coarse = np.asarray(coarse, dtype=np.float32)
    Nc = len(coarse)
    if Nc == 0:
        return np.zeros(fine_len, dtype=np.float32)
    if Nc == 1:
        return np.full(fine_len, coarse[0], dtype=np.float32)

    # 一旦2*Nc-1の等間隔に補間してから、必要に応じて切る/端埋め
    fine_est = np.empty(2 * Nc - 1, dtype=np.float32)
    fine_est[0::2] = coarse
    fine_est[1::2] = 0.5 * (coarse[:-1] + coarse[1:])
    # 長さ調整
    if fine_len <= len(fine_est):
        return fine_est[:fine_len].copy()
    else:
        # 足りない分は最後の値で埋める
        tail = np.full(fine_len - len(fine_est), fine_est[-1], dtype=np.float32)
        return np.concatenate([fine_est, tail])

def build_wx_pyramid(wx0, max_levels=None):
    """
    wxピラミッド（各レベルで 1/4 スケール）を作る
    """
    wx_levels = [wx0.astype(np.float32, copy=True)]
    L = 1
    while True:
        if max_levels is not None and L >= max_levels:
            break
        prev = wx_levels[-1]
        if len(prev) <= 4:
            break
        nxt = smooth_downsample_1d(prev) * 0.25  # 1/h^2 スケール
        wx_levels.append(nxt)
        L += 1
    return wx_levels

def v_cycle_1d(x, b, wx_levels, level=0, pre=2, post=2, omega=0.9, coarse_iters=30):
    """
    1D Vサイクル（誤差方程式の標準実装）
    - x: 現レベルの解ベクトル
    - b: 現レベルの右辺
    - wx_levels[level]: 現レベルの重み
    """
    wx = wx_levels[level]
    N = len(wx)
    # プレスムーズ
    x = jacobi_smooth_1d(x, b, wx, iters=pre, omega=omega)

    # 残差
    r = b - apply_A_1d(x, wx)

    # 収束 or 最粗レベル条件
    is_coarsest = (level == len(wx_levels) - 1) or (N <= 8)
    if is_coarsest:
        # 最粗レベルで r を右辺として A e = r を反復で近似解
        e = np.zeros_like(x)
        e = jacobi_smooth_1d(e, r, wx, iters=coarse_iters, omega=omega)
        x = x + e
        # ポストスムーズ（軽く）
        x = jacobi_smooth_1d(x, b, wx, iters=max(1, post//2), omega=omega)
        return x

    # 制限
    r_coarse = smooth_downsample_1d(r)
    # 粗レベルで誤差方程式 A e = r を解く
    x0 = np.zeros_like(r_coarse)
    e_coarse = v_cycle_1d(x0, r_coarse, wx_levels, level+1, pre=pre, post=post, omega=omega, coarse_iters=coarse_iters)
    # 延長して修正
    e_fine = prolong_linear_1d(e_coarse, N)
    x = x + e_fine

    # ポストスムーズ
    x = jacobi_smooth_1d(x, b, wx, iters=post, omega=omega)
    return x

def restore_1d_mg(pd, da, lam=100.0, threshold=1.0,
                  levels=None, cycles=1, pre=2, post=2, omega=0.9, coarse_iters=30,
                  clip_range=None, verbose=True):
    """
    1Dマルチグリッド解法（Vサイクル×cycles）
    - A x = b, A = I - div(wx ∂x·), wx = lam * W2
    """
    pd = np.asarray(pd, dtype=np.float32)
    da = np.asarray(da, dtype=np.float32)
    N = len(pd)
    assert len(da) == N
    _, W2 = make_weights_1d(da, threshold=threshold)
    wx0 = lam * np.clip(W2, 0.0, np.finfo(np.float32).max)

    # クリップ範囲
    if clip_range is None:
        vmin = float(min(pd.min(), da.min()))
        vmax = float(max(pd.max(), da.max()))
    else:
        vmin, vmax = clip_range

    # ピラミッド（wxのみでOK）
    wx_levels = build_wx_pyramid(wx0, max_levels=levels)

    # 右辺（最細のみ明示的に計算。粗レベルは残差制限で作る）
    b = compute_b_1d(pd, da, wx0)

    # 初期解（J1やJ2が無難。ここではpd）
    x = pd.copy()

    t0 = time.time()
    for c in range(cycles):
        x = v_cycle_1d(x, b, wx_levels, level=0, pre=pre, post=post, omega=omega, coarse_iters=coarse_iters)
        x = np.clip(x, vmin, vmax)
        if verbose:
            # 進捗表示（残差ノルム）
            res = b - apply_A_1d(x, wx0)
            rn = np.linalg.norm(res) / (np.sqrt(N) + 1e-12)
            print(f"V-cycle {c+1}/{cycles}: residual RMS = {rn:.6e}")
    if verbose:
        print(f"MG elapsed: {time.time() - t0:.3f} sec")
    return x

# ====== 1D 厳密解（Thomas法）※1D限定の最速オプション ======

def solve_tridiag_1d(pd, da, lam=100.0, threshold=1.0):
    """
    1DのA x = bを直接解く（Thomas法, O(N)）
    Aの3重対角は:
      diag[i]   = 1 + wE[i] + wW[i]
      lower[i]  = -wW[i]  (i>=1)
      upper[i]  = -wE[i]  (i<=N-2)
    Neumannに合わせて wW[0]=0, wE[N-1]=0
    """
    pd = np.asarray(pd, dtype=np.float32)
    da = np.asarray(da, dtype=np.float32)
    N = len(pd)
    _, W2 = make_weights_1d(da, threshold=threshold)
    wx = lam * np.clip(W2, 0.0, np.finfo(np.float32).max)
    # 係数
    wE = wx.copy(); wE[-1] = 0.0
    wW = np.empty_like(wx); wW[0] = 0.0; wW[1:] = wx[:-1]
    diag = 1.0 + wE + wW
    lower = -wW[1:]            # 長さ N-1
    upper = -wE[:-1]           # 長さ N-1
    b = compute_b_1d(pd, da, wx)

    # Thomas法（前進消去）
    c = upper.copy()
    d = b.copy()
    a = lower.copy()
    for i in range(1, N):
        m = a[i-1] / diag[i-1]
        diag[i] -= m * c[i-1]
        d[i]    -= m * d[i-1]
    # 後退代入
    x = np.empty(N, dtype=np.float32)
    x[-1] = d[-1] / diag[-1]
    for i in range(N-2, -1, -1):
        x[i] = (d[i] - c[i] * x[i+1]) / diag[i]
    return x

# ====== デモ（set_dataが無い環境用に簡易GT生成も用意） ======

def _toy_data(N=500, seed=0):
    rng = np.random.RandomState(seed)
    gt = np.zeros(N, dtype=np.float32)
    gt[N//4: N//2] = 1.0
    gt[N//2: 3*N//4] = 0.4
    da = gt * 1.05 + 0.02 * rng.randn(N).astype(np.float32)  # ガイド
    pd = gt * 0.9  + 0.05 * rng.randn(N).astype(np.float32)  # 観測
    return gt, pd, da

if __name__ == "__main__":
    try:
        from set_data import set_gt, set_pd, set_da
        N = 1000
        gt = set_gt(N); da = set_da(gt); pd = set_pd(gt)
    except Exception:
        gt, pd, da = _toy_data(2000)

    # マルチグリッド
    x_mg = restore_1d_mg(pd, da, lam=500.0, threshold=10.0,
                         levels=None, cycles=5, pre=2, post=2, omega=0.9, coarse_iters=40,
                         clip_range=None, verbose=True)

    # 厳密解（1Dだけで可）
    x_thomas = solve_tridiag_1d(pd, da, lam=50.0, threshold=10.0)

    mse_pd = np.mean((pd - gt)**2)
    mse_da = np.mean((da - gt)**2)
    mse_mg = np.mean((x_mg - gt)**2)
    mse_th = np.mean((x_thomas - gt)**2)

    print(f"MSE pd={mse_pd:.4f}, da={mse_da:.4f}, MG={mse_mg:.4f}, Thomas={mse_th:.4f}")

    # --- 結果の可視化 ---
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(gt, label='Ground Truth (gt)', linestyle='-', marker='o', markersize=2)
    plt.plot(pd, label='Stereo (pd)', linestyle='--', marker='o', markersize=2)
    plt.plot(da, label='Monocular depth (da)', linestyle='-.', marker='s', markersize=2)
    plt.plot(x_mg, label='Restored MG', linestyle='-', marker='x', markersize=2)
    plt.plot(x_thomas, label='Restored Thomas', linestyle=':', marker='.', markersize=2)
    plt.legend()
    plt.title('1D Depth Fusion Results')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.tight_layout()
    plt.show()
