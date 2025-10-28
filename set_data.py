import numpy as np

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
