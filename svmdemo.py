# ========================================
# svmdemo.py
# ========================================
# - scikit-learn の load_digits() を使うため、別途データダウンロード不要で実行できます
# - 0 vs 1 の 2 クラス分類を “一行で” SVM モデルを定義 → 学習 → 評価 しています

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def main():
    # 1) 手書き数字データ (8x8 ピクセル, 約1797 サンプル) を呼び出し
    digits = datasets.load_digits()
    X = digits.data      # shape = (n_samples, 64)  # 8x8 ピクセルをフラットにした 64 次元
    y = digits.target    # ラベルは 0～9 の数字

    # 2) 今回は “0 (数字の0) と 1 (数字の1)” の 2 クラスだけ抜き出す
    mask = (y == 0) | (y == 1)
    X2 = X[mask]
    y2 = y[mask]

    # 3) 訓練用 / テスト用に 70:30 で分割
    X_train, X_test, y_train, y_test = train_test_split(
        X2, y2, test_size=0.3, random_state=42, stratify=y2
    )

    # 4) SVM モデルを“1行”で定義して学習
    svm = SVC(kernel='linear', C=1.0, random_state=42)
    svm.fit(X_train, y_train)

    # 5) テストデータで予測し、精度を表示
    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"----- SVM (0 vs 1) テストセット正解率: {acc*100:.2f}% -----")

if __name__ == "__main__":
    main()
