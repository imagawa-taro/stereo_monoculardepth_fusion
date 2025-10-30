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
    # 1/10に縮小, データ数も削減
    pd = gt.copy()
    pd = np.convolve(pd, np.ones(10)/10, mode='same')
    pd = pd[::10]
    # 量子化
    pd = np.round(pd / 5.0) * 5.0
    # ノイズ
    pd += np.random.normal(0, 1.0, size=len(pd)).astype(np.float32)

    # 元のサイズにbiqubic補間拡大
    pd = np.interp(np.arange(len(gt)), np.linspace(0, len(gt)-1, num=len(pd)), pd)
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
