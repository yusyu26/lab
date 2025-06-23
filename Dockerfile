# ===========================================
# Dockerfile (bash 起動＋SVM 環境構築バージョン)
# ===========================================
# ベースイメージは Python 3.8-slim を利用
FROM python:3.8-slim

# バッファリングをオフ（任意）
ENV PYTHONUNBUFFERED=1

# apt のパッケージリスト更新および最低限のビルドツールをインストール
# （OpenCV を pip install するために gcc が必要になるケースがあるため入れておくと安心です）
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libglib2.0-0 \
        libgl1       \
        libsm6       \
        libxrender1  \
        libxext6     \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリを /app に設定
WORKDIR /app

# 必要な Python パッケージをまとめてインストール
# - numpy, scikit-learn : SVM を動かすために必須
# - matplotlib        : （後で可視化したい場合に使う。不要なら省いても OK）
# - opencv-python     : CASIA 等の実画像を扱いたいときに必要（MNIST だけなら不要ですが、念のため入れておきます）
RUN pip install --no-cache-dir \
        numpy \
        scikit-learn \
        scikit-image \
        matplotlib \
        opencv-python \
        torch \
        torchvision

# （もし後で GLCM などを試したい場合は scikit-image も追加してください）
# RUN pip install --no-cache-dir scikit-image

# ホスト側のコード／スクリプトをすべてコンテナ内の /app にコピーする
COPY . /app

# デフォルトのコマンドを Bash に変更
# これで docker run すると自動的に /bin/bash が立ち上がります
CMD ["bash"]
