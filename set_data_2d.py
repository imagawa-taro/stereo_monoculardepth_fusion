import numpy as np
from typing import Tuple, List, Optional

def set_gt_img(H: int, W: int) -> np.ndarray:
    """
    デモ用のGT画像を設定
    gt: 段差を含む画像
    """
    ww = W // 5
    hh = H // 5
    gt_img = np.zeros((H, W), dtype=np.float32)
    for i in range(ww):
        for j in range(ww):
            if (ww + j < H) and (ww + i < W):
                gt_img[hh + j, ww + i] = 30 + 15.0 * np.sin(np.pi * i / ww)  # 半円形
            if (2*ww+5 + j < H) and (3*ww + i < W):
                gt_img[2*hh+5 + j, 3*ww + i] = 30 + 15.0 * np.sin(np.pi * i / ww)  # 半円形
    return gt_img

def set_pd_img(gt_img: np.ndarray) -> np.ndarray:
    """
    デモ用の観測画像 pd を設定
    pd: gt をぼかし＋量子化＋ノイズ
    """
    np.random.seed(1)
    # 1/10に縮小, データ数も削減
    pd_img = gt_img.copy()
    kernel = np.ones((1, 10)) / 10
    pd_img = np.apply_along_axis(lambda m: np.convolve(m, kernel.flatten(), mode='same'), axis=1, arr=pd_img)
    pd_img = pd_img[:, ::10]
    # 量子化
    pd_img = np.round(pd_img / 5.0) * 5.0
    # ノイズ
    pd_img += np.random.normal(0, 1.0, size=pd_img.shape).astype(np.float32)

    # 元のサイズにbiqubic補間拡大
    x_old = np.linspace(0, pd_img.shape[1]-1, num=pd_img.shape[1])
    x_new = np.linspace(0, pd_img.shape[1]-1, num=gt_img.shape[1])
    pd_img_resized = np.empty_like(gt_img)
    for i in range(pd_img.shape[0]):
        pd_img_resized[i, :] = np.interp(x_new, x_old, pd_img[i, :])
    return pd_img_resized

def set_da_img(gt_img: np.ndarray) -> np.ndarray:
    """
    デモ用のガイド画像 da を設定
    da: gt の後ろ半分の段差を10低くする
    """
    da_img = gt_img.copy()
    H, W = da_img.shape
    da_img[:, W//2:] = np.maximum(da_img[:, W//2:] - 20, 0)
    return da_img
