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
from skimage.feature import graycomatrix, graycoprops

# --- グローバルスコープに関数を定義 ---

def load_casia_filepaths(casia_root_path):
    """指定されたパスからCASIAのファイルパスとラベルを取得する"""
    class_map = {'Au': 0, 'Tp': 1}
    filepaths = []
    
    for class_name, label_id in class_map.items():
        class_dir = os.path.join(casia_root_path, class_name)
        if not os.path.isdir(class_dir):
            print(f"[警告] ディレクトリが見つかりません: {class_dir}")
            continue

        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
                filepaths.append(os.path.join(class_dir, filename))
                
    return filepaths

def _calculate_glcm_features(channel_np):
    """
    単一のチャンネル画像からGLCM特徴量（16次元ベクトル）を計算するヘルパー関数
    """
    # Scharrフィルタを適用してエッジ画像を生成
    scharr_x = cv2.Scharr(channel_np, cv2.CV_64F, dx=1, dy=0)
    scharr_y = cv2.Scharr(channel_np, cv2.CV_64F, dx=0, dy=1)
    abs_scharr_x = cv2.convertScaleAbs(scharr_x)
    abs_scharr_y = cv2.convertScaleAbs(scharr_y)
    edges_np = cv2.addWeighted(abs_scharr_x, 0.5, abs_scharr_y, 0.5, 0)

    # GLCM特徴量を計算
    glcm_input = cv2.convertScaleAbs(edges_np)
    glcm = graycomatrix(glcm_input, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    
    contrast = graycoprops(glcm, 'contrast').flatten()
    dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()
    energy = graycoprops(glcm, 'energy').flatten()
    
    return np.concatenate([contrast, dissimilarity, homogeneity, energy])

def prepare_data_from_filepaths(filepaths, transform, channel_mode):
    """
    ファイルパスのリストから、指定されたモードで特徴量を抽出し、
    SVM用のデータ（X）とラベル（y）を作成する関数
    """
    X_data = []
    y_data = []

    for fp in filepaths:
        try:
            img = Image.open(fp)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img_resized = transform(img)
            img_np = np.array(img_resized)

            if img_np is None or img_np.ndim != 3 or img_np.shape[2] != 3:
                print(f"[警告] スキップ（画像形式エラー）: {fp}")
                continue

            ycrcb_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)

            if channel_mode == 'Cr':
                cr_channel_np = ycrcb_np[:, :, 1]
                feature_vector = _calculate_glcm_features(cr_channel_np)
            elif channel_mode == 'Cb':
                cb_channel_np = ycrcb_np[:, :, 2]
                feature_vector = _calculate_glcm_features(cb_channel_np)
            elif channel_mode == 'Cr+Cb':
                cr_channel_np = ycrcb_np[:, :, 1]
                cb_channel_np = ycrcb_np[:, :, 2]
                cr_features = _calculate_glcm_features(cr_channel_np)
                cb_features = _calculate_glcm_features(cb_channel_np)
                feature_vector = np.concatenate([cr_features, cb_features])
            else:
                raise ValueError("無効なchannel_modeです。'Cr', 'Cb', または 'Cr+Cb'を選択してください。")

            X_data.append(feature_vector)
            
            label = 1 if 'Tp' in os.path.basename(fp) else 0
            y_data.append(label)

        except Exception as e:
            print(f"[警告] スキップ（予期せぬエラー）: {fp}, エラー: {e}")
            continue

    return np.array(X_data), np.array(y_data)

# --- メインの処理 ---
def main():
    # ===============================================================
    # ===            ↓ この変数を書き換えて実験を切り替える ↓     ===
    # ===============================================================
    
    # 'Cr' または 'Cb' または 'Cr+Cb' を選択
    EXPERIMENT_MODE = 'Cr+Cb' 

    # ===============================================================

    CASIA_ROOT = './data/CASIA2'
    class_names = {0: 'Au (Original)', 1: 'Tp (Tampered)'}
    print(f"[CONFIG] 実験モード: {EXPERIMENT_MODE}")
    
    # 1) データセットの全ファイルパスを取得
    all_filepaths = load_casia_filepaths(CASIA_ROOT)
    if not all_filepaths:
        print("[エラー] 画像ファイルが見つかりませんでした。CASIA_ROOTのパスを確認してください。")
        return

    # 2) 訓練データとテストデータにファイルパスを分割
    random.seed(42)
    n_per_class_train = 100
    
    # ラベルごとにファイルパスを分類
    filepaths_au = [fp for fp in all_filepaths if 'Au' in os.path.basename(fp)]
    filepaths_tp = [fp for fp in all_filepaths if 'Tp' in os.path.basename(fp)]
    
    if len(filepaths_au) < n_per_class_train or len(filepaths_tp) < n_per_class_train:
        print(f"[エラー] 各クラスから{n_per_class_train}枚の画像を抽出できません。")
        return
        
    train_filepaths = random.sample(filepaths_au, n_per_class_train) + random.sample(filepaths_tp, n_per_class_train)
    test_filepaths = list(set(all_filepaths) - set(train_filepaths))
    
    # 3) 特徴抽出の準備
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
    ])

    # 4) 訓練データとテストデータの特徴量を抽出
    print("\n[INFO] 訓練データの特徴量抽出を開始します...")
    X_train, y_train = prepare_data_from_filepaths(train_filepaths, transform, EXPERIMENT_MODE)
    
    print("\n[INFO] テストデータの特徴量抽出を開始します...")
    X_test, y_test = prepare_data_from_filepaths(test_filepaths, transform, EXPERIMENT_MODE)

    print(f"\n[INFO] 訓練データ数: {len(X_train)}")
    print(f"[INFO] テストデータ数: {len(X_test)}")
    
    if len(X_train) == 0 or len(X_test) == 0:
        print("[エラー] 特徴量を抽出できませんでした。")
        return

    # 5) SVMモデルの学習と評価
    print("\n[INFO] SVMモデルの学習と評価を開始します...")
    svm = SVC(kernel='linear', C=1.0, random_state=42)
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n[RESULT] テストセット全体での正解率: {accuracy*100:.2f}%")

if __name__ == "__main__":
    main()