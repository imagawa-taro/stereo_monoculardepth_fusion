import torch
import numpy as np
from PIL import Image

def load_image(path):
    """
    16bit 画像を読み込み、FloatTensor 化 (1×1×H×W)
    """
    img = Image.open(path)
    arr = np.array(img, dtype=np.uint16)
    tensor = torch.from_numpy(arr.astype(np.float32))   # 0～65535 の float32
    tensor = tensor.unsqueeze(0).unsqueeze(0)           # shape = (1,1,H,W)
    return tensor

def compute_cost(x, f, W1, W2, W3, lam_x, lam_y):
    """
    C1: データフィデリティ項 (位置重み W1)
    C2: x 方向 2階差分滑らかさ項 (位置重み W2)
    C3: y 方向 2階差分滑らかさ項 (位置重み W3)
    """
    # --- C1 = mean( W1 * (x - f)^2 ) ---
    diff = (x - f)**2
    C1 = torch.mean(W1 * diff)

    # --- C2 = mean( W2[i,j] * (x[i,j+1] - 2 x[i,j] + x[i,j-1])^2 ) ---
    Dx2 = x[..., :, 2:] - 2.0 * x[..., :, 1:-1] + x[..., :, :-2]    # shape=(1,1,H,W-2)
    W2_crop = W2[..., :, 1:-1]                                     # 同じ shape になるよう中央を切り出し
    C2 = torch.mean(W2_crop * Dx2**2)

    # --- C3 = mean( W3[i,j] * (x[i+1,j] - 2 x[i,j] + x[i-1,j])^2 ) ---
    Dy2 = x[..., 2:, :] - 2.0 * x[..., 1:-1, :] + x[..., :-2, :]    # shape=(1,1,H-2,W)
    W3_crop = W3[..., 1:-1, :]                                     # 同じ shape に切り出し
    C3 = torch.mean(W3_crop * Dy2**2)

    C = C1 + lam_x * C2 + lam_y * C3
    return C1, C2, C3, C

def main():
    # --- デバイス設定 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 観測画像 f の読み込み ---
    f = load_image('P1020116c_depth16.png').to(device)  # shape=(1,1,3000,2000)

    # --- 位置重み W1, W2, W3 の初期化 ---
    # 仮に全て 1（有効化）で初期化。将来的にはファイル読み込みや位置依存マップを設定。
    W1 = torch.ones_like(f, device=device)  # データ項用
    W2 = torch.ones_like(f, device=device)  # x方向2階差分用
    W3 = torch.ones_like(f, device=device)  # y方向2階差分用

    # --- 復元対象 x の初期値 ---
    x = f.clone().detach().to(device)  # 初期値は観測画像 f と同じ
    x.requires_grad = True
    # x = torch.zeros_like(f, device=device, requires_grad=True)
    # → 将来は別の初期マップを設定できるようにここを書き換え

    # --- ハイパーパラメータ ---
    lam_x     = 100000    # x 方向 2階差分の重み
    lam_y     = 100000    # y 方向 2階差分の重み
    lr        = 10.0    # 学習率
    num_iters = 3000

    # --- 最適化器 ---
    optimizer = torch.optim.Adam([x], lr=lr)

    # --- 反復最適化ループ ---
    for it in range(1, num_iters + 1):
        optimizer.zero_grad()
        C1, C2, C3, C = compute_cost(x, f, W1, W2, W3, lam_x, lam_y)
        C.backward()
        optimizer.step()

        # 画素値を 0〜65535 にクリッピング
        with torch.no_grad():
            x.clamp_(0, 65535)

        # ログ出力
        if it == 1 or it % 50 == 0:
            print(f"iter {it:4d} | C={C.item():.4f} | C1={C1.item():.4f} | C2={C2.item():.4f} | C3={C3.item():.4f}")

    # --- 16bit 画像として保存 ---
    out_arr = x.detach().cpu().numpy().astype(np.uint16).squeeze()
    Image.fromarray(out_arr).save('restored16bit_weighted.png')

if __name__ == '__main__':
    main()
    