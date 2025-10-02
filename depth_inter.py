import torch
import numpy as np
from PIL import Image
import time
from PIL import ImageFilter

def compute_cost2(x, pd, da, W1, W2, W3, lam_x, lam_y):
    """
    C1: データフィデリティ項 (位置重み W1)
    C2: X 方向 x差分とda差分の差 (位置重み W2)
    C3: Y 方向 x差分とda差分の差 (位置重み W3)
    """
    # --- C1 = mean( W1 * (x - f)^2 ) ---
    diff = (x - pd)**2
    C1 = torch.mean(W1 * diff)

    # --- C2 = mean( W2[i,j] * ((x[i,j+1] -  x[i,j])-(da[i,j+1] -  da[i,j]))^2 ) ---
    Dx2 = (x[..., :, 1:] - x[..., :, :-1]) - (da[..., :, 1:] - da[..., :, :-1])  # shape=(1,1,H,W-1)
    W2_crop = W2[..., :, 1:]                                     # 同じ shape になるよう中央を切り出し
    C2 = torch.mean(W2_crop * Dx2**2)

    # --- C3 = mean( W3[i,j] * ((x[i+1,j] -  x[i,j])-(da[i+1,j] -  da[i,j]))^2 ) ---
    Dy2 = (x[..., 1:, :] - x[..., :-1, :]) - (da[..., 1:, :] - da[..., :-1, :])  # shape=(1,1,H-1,W)
    W3_crop = W3[..., 1:, :]                                     # 同じ shape に切り出し
    C3 = torch.mean(W3_crop * Dy2**2)

    C = C1 + lam_x * C2 + lam_y * C3
    return C1, C2, C3, C


def main():
    # --- デバイス設定 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 参照画像の読み込み ---
    folder_path = 'image/3persons/'
    da_image = Image.open(folder_path + 'F4_L_depth16_small.png') # 単眼デプス
    pd_image = Image.open(folder_path + 'PD_disparity10.png') # ステレオ視差
    gt_image = Image.open(folder_path + 'Depth_cm.png') # 真値デプス
    img_array = np.array(gt_image) * 10
    gt_image = Image.fromarray(img_array.astype(np.uint16))  # 画像の各ピクセル値を10倍
    # pd_imageノイズ除去: medianフィルタ
    pd_image = pd_image.filter(ImageFilter.MedianFilter(size=15))
    # 画像サイズを統一
    pd_image = pd_image.resize(da_image.size, Image.LANCZOS)
    # スケール調整
    # pd_imageをgt_imageの最大値・最小値で正規化
    pd_array = np.array(pd_image, dtype=np.float32)
    gt_array = np.array(gt_image, dtype=np.float32)
    pd_array = (pd_array - pd_array.min()) / (pd_array.max() - pd_array.min()) * (gt_array.max() - gt_array.min()) + gt_array.min()
    pd_array = gt_array.max() - pd_array
    # pd_imageを保存
    pd_image = Image.fromarray(pd_array.astype(np.uint16))
    pd_image.save('pd_image.png')
    # da_imageをpd_imageの最大値・最小値で正規化
    da_array = np.array(da_image, dtype=np.float32)
    da_array = (da_array - da_array.min()) / (da_array.max() - da_array.min()) * (pd_array.max() - pd_array.min()) + pd_array.min()

    # --- 位置重み W1, W2, W3 の計算 ---
    # pdのx差分、y差分を計算し、差分がthresholdより大きいところはW2,W3を小さくする
    W2 = np.ones_like(pd_array, dtype=np.float32)  # データ項用
    pd_dx = np.abs(np.diff(pd_array, axis=1, prepend=pd_array[:, :1]))  # x方向差分
    pd_dy = np.abs(np.diff(pd_array, axis=0, prepend=pd_array[:1, :]))  # y方向差分
    # pd_dx, pd_dyが大きいところはW2,W3を小さくする
    threshold = 1
    W2[pd_dx > threshold] = 0.01
    W3 = W2.copy()

    # Tensor化
    da = torch.from_numpy(da_array).unsqueeze(0).unsqueeze(0).to(device)  # shape=(1,1,H,W)
    pd = torch.from_numpy(pd_array).unsqueeze(0).unsqueeze(0).to(device)  # shape=(1,1,H,W)
    W1 = torch.from_numpy(np.ones_like(pd_array, dtype=np.float32)).unsqueeze(0).unsqueeze(0).to(device)  # shape=(1,1,H,W)
    W2 = torch.from_numpy(W2).unsqueeze(0).unsqueeze(0).to(device)  # shape=(1,1,H,W)
    W3 = torch.from_numpy(W3).unsqueeze(0).unsqueeze(0).to(device)  # shape=(1,1,H,W)

    # --- 復元対象 x の初期値 ---
    x = da.clone().detach().to(device)  # 初期値は観測画像da
    x.requires_grad = True

    # --- ハイパーパラメータ ---
    lam_x     = 10    # x 方向 1階差分の重み
    lam_y     = 10    # y 方向 1階差分の重み
    lr        = 100.0    # 学習率
    num_iters = 1000

    # --- 最適化器 ---
    optimizer = torch.optim.Adam([x], lr=lr)

    # --- 反復最適化ループ ---
    time1 = time.time()
    for it in range(1, num_iters + 1):
        optimizer.zero_grad()
        C1, C2, C3, C = compute_cost2(x, pd, da, W1, W2, W3, lam_x, lam_y)
        C.backward()
        optimizer.step()

        # 画素値を 0〜65535 にクリッピング
        with torch.no_grad():
            x.clamp_(0, 65535)

        # ログ出力
        if it == 1 or it % 50 == 0:
            print(f"iter {it:4d} | C={C.item():.4f} | C1={C1.item():.4f} | C2={C2.item():.4f} | C3={C3.item():.4f}")

    print('Elapsed time: %.1f sec' % (time.time() - time1))

    # --- 16bit 画像として保存 ---
    out_arr = x.detach().cpu().numpy().astype(np.uint16).squeeze()
    Image.fromarray(out_arr).save('restored.png')

if __name__ == '__main__':
    main()
    