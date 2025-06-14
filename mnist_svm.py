import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def main():
    target_labels = [4, 9]

    label_a, label_b = target_labels
    print(f"[CONFIG] 分類クラス: {label_a} vs {label_b}")

    # 1) MNISTデータセットをロード
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 訓練データとテストデータをダウンロード
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # 2) 指定したラベルのみ抽出
    # 訓練データから指定したラベルのインデックスを抽出
    train_indices_a = [i for i, (_, label) in enumerate(train_dataset) if label == label_a]
    train_indices_b = [i for i, (_, label) in enumerate(train_dataset) if label == label_b]

    print(f"[INFO] 訓練データ内の数字{label_a}のサンプル数: {len(train_indices_a)}")
    print(f"[INFO] 訓練データ内の数字{label_b}のサンプル数: {len(train_indices_b)}")

    # テストデータから指定したラベルのみを抽出
    test_indices = [i for i, (_, label) in enumerate(test_dataset) if label in target_labels]
    print(f"[INFO] テストデータ内の{label_a}と{label_b}のサンプル数: {len(test_indices)}")

    # 3) 訓練用に「各クラス5枚ずつ」だけ使う (seed=42で固定)
    random.seed(42)
    n_per_class = 5
    train_indices = (
        random.sample(train_indices_a, n_per_class) +
        random.sample(train_indices_b, n_per_class)
    )

    # 選択したインデックスからデータを取得
    X_train = []
    y_train = []
    for idx in train_indices:
        img, label = train_dataset[idx]
        # PyTorchのテンソルをnumpyに変換し、1次元に平坦化
        X_train.append(img.numpy().flatten())
        y_train.append(label)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # テストデータも同様に準備
    X_test = []
    y_test = []
    test_img_tensors = []  # 後で元の画像を表示するために保持

    for idx in test_indices:
        img, label = test_dataset[idx]
        X_test.append(img.numpy().flatten())
        y_test.append(label)
        test_img_tensors.append(img)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    print(f"[INFO] トレーニングに使う{label_a} vs {label_b}の枚数: {len(train_indices)} （各クラス{n_per_class}枚ずつ）")
    print(f"[INFO] テスト用サンプル数: {len(test_indices)}")

    # 4) SVMモデルを定義・学習・評価
    svm = SVC(kernel='linear', C=1.0, random_state=0)
    svm.fit(X_train, y_train)

    y_pred_all = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred_all)
    print(f"[RESULT] テストセット全体での正解率: {acc*100:.2f}%")

    # 5) テストセットから「毎回ランダムに」1枚選び、表示＆保存
    random.seed(None)  # seedをリセット
    pick_idx = random.randrange(len(test_indices))

    sample_vec = X_test[pick_idx]
    sample_true = y_test[pick_idx]
    sample_img_tensor = test_img_tensors[pick_idx]

    sample_pred = svm.predict(sample_vec.reshape(1, -1))[0]

    print(f"[SAMPLE] 選択インデックス: {pick_idx}")
    print(f"         → 真ラベル: {sample_true}, 予測ラベル: {sample_pred}")

    # Matplotlibで可視化して、PNGファイルとして保存
    plt.figure(figsize=(4, 4))
    plt.imshow(sample_img_tensor[0], cmap='gray')  # 最初のチャンネルを取得
    plt.title(f"True: {sample_true} / Pred: {sample_pred}", fontsize=14)
    plt.axis('off')

    save_path = "output/mnist_prediction.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"[INFO] 予測サンプル画像を '{save_path}' として保存しました")
    plt.show()

if __name__ == "__main__":
    main()