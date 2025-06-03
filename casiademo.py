# ============================================
# casia_interactive.py
# ============================================
#
# ・CASIA2.0 の dataset/Au（オリジナル）と dataset/Tp（改ざん）を使い、
#   学習時には「改ざん画像の枚数に合わせてオリジナルをランダムサンプリング」してバランスをとります。
# ・一度学習が終わると、モデルとテストセットを pickle で保存し（svm_model.pkl, test_data.pkl）、
#   それ以降の実行では「学習済みモデルを読み込み → ランダムに１枚テスト画像を選択 → 予測結果と画像を表示」
#   の部分だけを繰り返し実行します。
#
# 実行手順：
# 1) 初回：dataset/Au, dataset/Tp フォルダを用意し、以下のコマンドを実行するとバランス学習が行われます。
#       python casia_interactive.py
#    このとき、svm_model.pkl, test_data.pkl が生成されます。
#
# 2) 2回目以降：すでに svm_model.pkl と test_data.pkl が存在する場合は、
#       python casia_interactive.py
#    と実行すると「ランダムに１枚ずつテスト画像を選んで予測 → 結果と画像を表示」を繰り返します。
#    （Enter キーを押すたびに次のサンプルを表示します。q を入力すれば終了します）
#
# 必要なパッケージ：
#    numpy, scikit-learn, matplotlib, opencv-python, pickle (標準ライブラリ)
#

import os
import random
import pickle

import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def load_casia_binary(folder_path, label, size=(64, 64)):
    """
    指定フォルダ内の画像をすべて読み込み、グレースケール→リサイズ→flatten したベクトルを返す。
    """
    X_list = []
    y_list = []
    for fname in os.listdir(folder_path):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue
        path = os.path.join(folder_path, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img_resized = cv2.resize(img, size)
        vec = img_resized.flatten()  # (4096,)
        X_list.append(vec)
        y_list.append(label)
    return X_list, y_list


def train_and_save_model():
    """
    ・dataset/Au と dataset/Tp フォルダから全画像を読み込み
    ・改ざん (Tp) の枚数に合わせてオリジナル (Au) をランダムサンプリングし、
      バランスの取れた train データを作成
    ・残りを test データとし、線形 SVM を学習して精度を表示
    ・学習済みモデルを svm_model.pkl に保存
    ・test セット (X_test, y_test) を test_data.pkl に保存
    """
    dataset_root = "CASIA2"
    au_folder = os.path.join(dataset_root, "Au")
    tp_folder = os.path.join(dataset_root, "Tp")

    # 1) 画像を読み込んでベクトル化 (Au→0, Tp→1)
    print("=== 画像を読み込んでフラットベクトル化します ===")
    X_au, y_au = load_casia_binary(au_folder, label=0, size=(64, 64))
    X_tp, y_tp = load_casia_binary(tp_folder, label=1, size=(64, 64))

    n_au = len(y_au)
    n_tp = len(y_tp)
    print(f"[INFO] Au (オリジナル) 画像数: {n_au}")
    print(f"[INFO] Tp (改ざん)     画像数: {n_tp}")

    # 2) 改ざん画像枚数に合わせてオリジナルをランダムサンプル
    #    (seed=None で毎回異なるサンプルにする。変更したければ固定値にする)
    random.seed(None)
    if n_tp < n_au:
        sampled_indices = random.sample(range(n_au), n_tp)
        X_au_sample = [X_au[i] for i in sampled_indices]
        y_au_sample = [y_au[i] for i in sampled_indices]
    else:
        # 万一改ざん画像が多い場合はそのまま全部
        X_au_sample = X_au[:]
        y_au_sample = y_au[:]

    # 改ざん側もすべて使う
    X_tp_sample = X_tp[:]
    y_tp_sample = y_tp[:]

    # 3) バランスのとれた train データを作成
    X_train = np.vstack([X_au_sample, X_tp_sample])
    y_train = np.array(y_au_sample + y_tp_sample, dtype=np.int32)

    # 4) 残りを test データとする
    #    オリジナル側の残り
    au_indices_all = set(range(n_au))
    au_sampled_set = set(sampled_indices if n_tp < n_au else [])
    au_rest_indices = sorted(list(au_indices_all - au_sampled_set))
    X_au_rest = [X_au[i] for i in au_rest_indices]
    y_au_rest = [y_au[i] for i in au_rest_indices]

    # 改ざん側は train で全て使っているので test には何も残らない
    # （=> test はオリジナルの残りのみ。もし改ざん側の一部も test に回したいならここを調整）
    X_test = np.vstack(X_au_rest) if len(X_au_rest) > 0 else np.empty((0, 64*64))
    y_test = np.array(y_au_rest, dtype=np.int32) if len(y_au_rest) > 0 else np.empty((0,))

    print(f"[INFO] バランス学習用トレーニングサンプル数: {len(y_train)}  (各クラス {len(y_tp_sample)} 枚ずつ)")
    print(f"[INFO] テスト用サンプル数: {len(y_test)} (すべてオリジナル画像)")

    # 5) 線形カーネル SVM を学習
    print("=== 線形カーネル SVM の学習を行います ===")
    svm = SVC(kernel="linear", C=1.0, random_state=0)
    svm.fit(X_train, y_train)

    # 6) テストセット全体で予測＆精度を表示
    print("=== テストセット全体で予測 & 正答率を計算します ===")
    if len(y_test) > 0:
        y_pred_all = svm.predict(X_test)
        acc_all = accuracy_score(y_test, y_pred_all)
        print(f"[RESULT] テストセット全体の正答率: {acc_all*100:.2f}%")
    else:
        print("[WARN] テスト用サンプルが存在しません (all data was used for training)")

    # 7) 学習済みモデルと テストデータ を pickle で保存
    with open("svm_model.pkl", "wb") as f_model:
        pickle.dump(svm, f_model)
    with open("test_data.pkl", "wb") as f_test:
        pickle.dump((X_test, y_test), f_test)
    print("[INFO] 学習済みモデルを 'svm_model.pkl' に保存しました")
    print("[INFO] テストデータを 'test_data.pkl' に保存しました")
    print("=== 学習フェーズ完了 ===\n")


def interactive_test():
    """
    ・svm_model.pkl を読み込み、test_data.pkl （X_test, y_test）を読み込み
    ・Enter キーを押すたびに「ランダムに1枚テスト画像を選んで予測 → 結果と画像を表示」を繰り返す
    ・"q" を入力すると終了
    """
    # 1) pickle ファイルを読み込む
    print("=== 学習済みモデルとテストデータを読み込みます ===")
    with open("svm_model.pkl", "rb") as f_model:
        svm = pickle.load(f_model)
    with open("test_data.pkl", "rb") as f_test:
        X_test, y_test = pickle.load(f_test)

    n_test = len(y_test)
    if n_test == 0:
        print("[ERROR] テストデータが存在しません。まずは学習フェーズを実行してください。")
        return

    print(f"[INFO] テストデータ枚数: {n_test}")
    print("Enterを押すとランダムに1枚を選択して予測 → 結果と画像を表示します")
    print("終了したいときは 'q' を入力して Enter を押してください\n")

    while True:
        cmd = input(">>> ")
        if cmd.lower().strip() == "q":
            print("=== 終了します ===")
            break

        # ランダムに1枚選ぶ
        idx = random.randrange(n_test)
        sample_vec  = X_test[idx]
        sample_true = y_test[idx]
        sample_pred = svm.predict(sample_vec.reshape(1, -1))[0]
        sample_img  = sample_vec.reshape((64, 64))

        # 結果を表示
        result_text = "正解" if sample_true == sample_pred else "不正解"
        print(f"[SAMPLE] テスト中のインデックス: {idx}  (真ラベル: {sample_true}, 予測: {sample_pred}) → {result_text}")

        # 画像を表示 & 保存
        plt.figure(figsize=(4, 4))
        plt.imshow(sample_img, cmap="gray")
        plt.title(f"True: {sample_true} / Pred: {sample_pred} → {result_text}", fontsize=14)
        plt.axis("off")

        save_path = f"sample_{idx}_result.png"
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[INFO] 画像を '{save_path}' に保存しました")
        plt.show()

    print()

if __name__ == "__main__":
    # svm_model.pkl が存在しなければ「学習フェーズ」を実行し、
    # 存在すれば「対話型テスト」を開始する
    if not os.path.exists("svm_model.pkl") or not os.path.exists("test_data.pkl"):
        train_and_save_model()
    else:
        interactive_test()
