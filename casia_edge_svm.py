import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import cv2

def load_casia_filepaths(casia_root_path):
    """指定されたパスからCASIAのファイルパスとラベルを取得する"""
    class_map = {'Au': 0, 'Tp': 1} # Au:オリジナル(0), Tp:改竄(1)
    filepaths = []
    labels = []

    for class_name, label_id in class_map.items():
        class_dir = os.path.join(casia_root_path, class_name)
        if not os.path.isdir(class_dir):
            print(f"[警告] ディレクトリが見つかりません: {class_dir}")
            continue

        for filename in os.listdir(class_dir):
            # サポートされている画像形式かチェック（必要に応じて拡張子を追加）
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
                filepaths.append(os.path.join(class_dir, filename))
                labels.append(label_id)

    return filepaths, labels

def main():
    CASIA_ROOT = './data/CASIA2'

    class_names = {0: 'Au (Original)', 1: 'Tp (Tampered)'}
    print(f"[CONFIG] 分類クラス: {class_names[0]} vs {class_names[1]}")

    # 1) CASIAデータセットのファイルパスをロード
    all_filepaths, all_labels = load_casia_filepaths(CASIA_ROOT)
    if not all_filepaths:
        print("[エラー] 画像ファイルが見つかりませんでした。CASIA_ROOTのパスを確認してください。")
        return

    # 前処理の定義（リサイズ、グレースケール化、テンソル化）
    transform = transforms.Compose([
        transforms.Resize((128, 128)), # 画像サイズを128x128に統一
    ])

    # 2) ラベルごとにファイルパスを分割
    filepaths_a = [fp for fp, label in zip(all_filepaths, all_labels) if label == 0]
    filepaths_b = [fp for fp, label in zip(all_filepaths, all_labels) if label == 1]

    print(f"[INFO] データ内の '{class_names[0]}' のサンプル数: {len(filepaths_a)}")
    print(f"[INFO] データ内の '{class_names[1]}' のサンプル数: {len(filepaths_b)}")

    # 3) 訓練用に「各クラス100枚ずつ」をランダムに抽出 (seed=42で固定)
    random.seed(42)
    n_per_class_train = 100 # 訓練に使う各クラスの枚数

    # データ数が足りるかチェック
    if len(filepaths_a) < n_per_class_train or len(filepaths_b) < n_per_class_train:
        print(f"[エラー] 各クラスから{n_per_class_train}枚の画像を抽出できません。枚数を減らしてください。")
        return

    train_filepaths_a = random.sample(filepaths_a, n_per_class_train)
    train_filepaths_b = random.sample(filepaths_b, n_per_class_train)
    train_filepaths = train_filepaths_a + train_filepaths_b

    # 訓練に使わなかった残りの全ファイルをテスト用とする
    test_filepaths = list(set(all_filepaths) - set(train_filepaths))

    # --- 変更点: ファイルパスから画像を読み込み、前処理を行う ---
    def prepare_data_from_filepaths(filepaths, transform):
        X_data = []
        y_data = []
        img_tensors = []

        for fp in filepaths:
            try:
                # Pillowで画像を開き、RGBに変換
                img = Image.open(fp).convert('RGB')

                # ステップ2で定義したリサイズ処理を適用
                img_resized = transform(img)

                # --- ここからがエッジ検出処理 ---

                # 1. Pillow ImageをOpenCVが扱えるNumpy配列に変換
                img_np = np.array(img_resized)

                # 2. OpenCVでグレースケールに変換 (RGB -> GRAY)
                gray_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

                # 変更後のコード：Scharrフィルタを適用
                # 1. 横方向のエッジを検出
                scharr_x = cv2.Scharr(gray_np, cv2.CV_64F, dx=1, dy=0)
                # 2. 縦方向のエッジを検出
                scharr_y = cv2.Scharr(gray_np, cv2.CV_64F, dx=0, dy=1)

                # 3. 横方向と縦方向のエッジを合成
                #    絶対値を取ってから加算することで、より明確なエッジ画像を得る
                abs_scharr_x = cv2.convertScaleAbs(scharr_x)
                abs_scharr_y = cv2.convertScaleAbs(scharr_y)
                edges_np = cv2.addWeighted(abs_scharr_x, 0.5, abs_scharr_y, 0.5, 0)

                # --- エッジ検出処理ここまで ---

                # 4. 処理後のNumpy配列をPyTorchのテンソルに変換
                #    グレースケールなのでチャンネル次元を追加 (unsqueeze(0))
                img_tensor = torch.from_numpy(edges_np).float().unsqueeze(0)

                # 5. ベクトル化してSVMへの入力データを作成
                X_data.append(img_tensor.numpy().flatten())

                # ファイルパスからラベルを判断 (Au or Tp)
                label = 1 if 'Tp' in os.path.basename(fp) else 0
                y_data.append(label)

                # 表示用に、処理後のテンソルも保存しておく
                img_tensors.append(img_tensor)

            except Exception as e:
                print(f"[警告] ファイル読み込みエラー: {fp}, エラー: {e}")
                continue

        return np.array(X_data), np.array(y_data), img_tensors

    X_train, y_train, _ = prepare_data_from_filepaths(train_filepaths, transform)
    X_test, y_test, test_img_tensors = prepare_data_from_filepaths(test_filepaths, transform)

    print(f"[INFO] トレーニングに使う枚数: {len(X_train)} （各クラス{n_per_class_train}枚ずつ）")
    print(f"[INFO] テスト用サンプル数: {len(X_test)}")
    if len(X_train) == 0 or len(X_test) == 0:
        print("[エラー] 訓練データまたはテストデータを作成できませんでした。")
        return

    # 4) SVMモデルを定義・学習・評価
    svm = SVC(kernel='linear', C=1.0, random_state=0)
    svm.fit(X_train, y_train)

    y_pred_all = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred_all)
    print(f"[RESULT] テストセット全体での正解率: {acc*100:.2f}%")

    # 5) テストセットから「毎回ランダムに」1枚選び、表示＆保存
    if not test_img_tensors:
        print("[INFO] 表示できるテストサンプルがありません。")
        return

    random.seed(None)  # seedをリセット
    pick_idx = random.randrange(len(X_test))

    sample_vec = X_test[pick_idx]
    sample_true_label = y_test[pick_idx]
    sample_img_tensor = test_img_tensors[pick_idx]

    sample_pred_label = svm.predict(sample_vec.reshape(1, -1))[0]

    sample_true_name = class_names[sample_true_label]
    sample_pred_name = class_names[sample_pred_label]

    print(f"[SAMPLE] 選択インデックス: {pick_idx}")
    print(f"         → 真ラベル: {sample_true_name}, 予測ラベル: {sample_pred_name}")

    # Matplotlibで可視化して、PNGファイルとして保存
    plt.figure(figsize=(4, 4))
    plt.imshow(sample_img_tensor[0], cmap='gray')
    plt.title(f"True: {sample_true_name}\nPred: {sample_pred_name}", fontsize=12)
    plt.axis('off')

    # 保存用ディレクトリを作成
    os.makedirs("output", exist_ok=True)
    save_path = "output/casia_edge_prediction.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"[INFO] 予測サンプル画像を '{save_path}' として保存しました")
    plt.show()

if __name__ == "__main__":
    main()