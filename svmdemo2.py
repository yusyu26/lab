# ============================================
# verbose_svm_random.py
# ============================================
# - “0 vs 1” の手書き数字データを使い、
#   訓練サンプル (各クラス 5 枚ずつ) は同じままに固定
# - 学習後、テストセットからランダムに１枚だけ選ぶ際に
#   random.seed(None) を使い、毎回異なるサンプルを表示・保存する

import os
import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def main():
    # 1) 手書き数字データを読み込み、"0 vs 1" のサンプルのみ抽出
    digits = datasets.load_digits()
    X_all = digits.data   # shape = (1797, 64)
    y_all = digits.target # 0〜9

    # "0 と 1 のみ" を抽出
    mask01 = (y_all == 0) | (y_all == 1)
    X01 = X_all[mask01]   # shape = (サンプル数, 64)
    y01 = y_all[mask01]   # 0 or 1

    print(f"[INFO] 全体サンプル (0 or 1) の枚数: {X01.shape[0]}")

    # 2) 訓練用に「各クラス 5 枚ずつ」だけ使う (seed=42 で固定)
    idx_class0 = np.where(y01 == 0)[0].tolist()  # 数字 0 のインデックスリスト
    idx_class1 = np.where(y01 == 1)[0].tolist()  # 数字 1 のインデックスリスト

    random.seed(42)
    n_per_class = 5
    train_idx = (
        random.sample(idx_class0, n_per_class) +
        random.sample(idx_class1, n_per_class)
    )
    train_idx = sorted(train_idx)

    all_indices = set(range(len(X01)))
    train_set = set(train_idx)
    test_idx = sorted(list(all_indices - train_set))

    X_train = X01[train_idx]
    y_train = y01[train_idx]
    X_test  = X01[test_idx]
    y_test  = y01[test_idx]

    print(f"[INFO] トレーニングに使う 0 vs 1 の枚数: {len(train_idx)} （各クラス {n_per_class} 枚ずつ）")
    print(f"[INFO] テスト用サンプル数: {len(test_idx)}")

    # 3) SVM モデルを定義・学習・評価
    svm = SVC(kernel='linear', C=1.0, random_state=0)
    svm.fit(X_train, y_train)

    y_pred_all = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred_all)
    print(f"[RESULT] テストセット全体での正解率: {acc*100:.2f}%")

    # 4) テストセットから「毎回ランダムに」1 枚選び、表示＆保存
    # ここだけ seed をリセットし、毎回異なる結果に
    random.seed(None)
    pick_i = random.choice(test_idx)  # X01, y01 のインデックスから選ぶ
    sample_vec  = X01[pick_i]
    sample_true = y01[pick_i]
    sample_image = sample_vec.reshape((8, 8))  # 8×8 ピクセルに復元

    sample_pred = svm.predict(sample_vec.reshape(1, -1))[0]

    print(f"[SAMPLE] 選択インデックス: {pick_i}")
    print(f"         → 真ラベル: {sample_true}, 予測ラベル: {sample_pred}")

    # Matplotlib で可視化して、PNG ファイルとして保存
    plt.figure(figsize=(3,3))
    plt.imshow(sample_image, cmap='gray')
    plt.title(f"True: {sample_true} / Pred: {sample_pred}", fontsize=14)
    plt.axis('off')

    save_path = "example_prediction.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"[INFO] 予測サンプル画像を '{save_path}' として保存しました")
    plt.show()

if __name__ == "__main__":
    main()
