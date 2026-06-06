# Conditional-Source Flow Matching for Semantic Segmentation

PyTorchによる，画像条件付きsource分布を用いたFlow Matchingベースのセマンティックセグメンテーション研究コードです．

## 概要

本リポジトリは，セマンティックセグメンテーションを生成問題として扱い，Flow Matchingによってセグメンテーションマスクを生成するための研究実装です．

カテゴリラベルとして表されるセグメンテーションマスクを，open simplex上のカテゴリ確率表現に変換し，さらにILR変換によってEuclidean空間へ写像します．その空間上で，入力画像に条件付けられたsource generatorにより初期分布をモデル化し，Flow Matchingの速度場を学習します．

本リポジトリは，学習設定，モデル構築，checkpointの読み込み，評価処理を整理し，再現しやすい研究コードとして公開することを目的としています．

## 背景

一般的な識別的セグメンテーション手法は，入力画像からラベルマップを直接予測します．一方，生成モデルとしてセグメンテーションを扱うことで，構造をもつ出力分布や不確実性を表現できる可能性があります．

本実装では，Flow Matchingにおけるsource分布を入力画像で条件付けることで，条件と無関係な初期分布から生成を始める場合よりも，セグメンテーションマスクに近い初期状態を与えられるかを検討します．

## 手法の概要

## 手法の概要

本実装では，セマンティックセグメンテーションをカテゴリラベルの直接予測ではなく，カテゴリ確率表現の生成問題として扱う．処理の流れは以下の通りである．

まず，one-hot表現のセマンティックマスク $y$ を，Dirichlet noiseによってopen simplex上の確率表現 $x_1$ に変換する．

$$
x_1 = \lambda y + (1 - \lambda)\epsilon,
\quad
\epsilon \sim \mathrm{Dirichlet}(\alpha_1 \mathbf{1})
$$

ここで，$\lambda$ はone-hotマスクとDirichlet noiseの混合比を表す．この処理により，カテゴリラベルを厳密なone-hot表現ではなく，すべての成分が正の確率ベクトルとして扱う．

次に，$x_1$ を対数空間に写し，ヘルマート行列に基づく直交コントラスト行列 $H \in \mathbb{R}^{(K-1)\times K}$ を用いて，Euclidean空間の表現 $z_1$ に変換する．

$$
z_1 = H \log x_1
$$

ここで，$K$ はクラス数であり，$H$ は各行が互いに直交し，かつ全要素和が0となるコントラスト基底を与える行列である．本実装では，`scipy.linalg.helmert(K, full=False)` により作成されるHelmert contrast matrixを用いる．この変換により，simplex上のカテゴリ確率を $K-1$ 次元の連続表現として扱うことができる．

source generatorは，入力画像 $I$ からsource分布の平均 $\mu_\theta(I)$ を予測する．固定分散を用いる場合，source sample $z_0$ は以下で表される．

$$
z_0 = \mu_\theta(I) + \sigma \epsilon,
\quad
\epsilon \sim \mathcal{N}(0, I)
$$

ここで，$\sigma$ はsource分布の標準偏差である．分散を学習する設定では，source generatorは平均に加えて対数分散も出力する．

Flow Matchingでは，source表現 $z_0$ からtarget表現 $z_1$ への経路上の点 $z_t$ を用いる．一般には，時刻変換 $\tau(t)$ を用いて以下のように表す．

$$
z_t = (1 - \tau(t))z_0 + \tau(t)z_1
$$

このとき，教師速度場は以下で与えられる．

$$
u_t = \dot{\tau}(t)(z_1 - z_0)
$$

本実験の標準設定では，$\tau(t)=t$ の線形経路を用いる．この場合，経路と教師速度場は以下のようになる．

$$
z_t = (1 - t)z_0 + tz_1
$$

$$
u_t = z_1 - z_0
$$

モデル $v_\theta$ は，入力画像 $I$，時刻 $t$，中間表現 $z_t$ を条件として速度場を予測する．Flow Matchingの速度場損失は以下で表される．

$$
\mathcal{L}_{\mathrm{FM}}
=========================

\mathbb{E}*{t, z_0, z_1}
\left[
\left|
v*\theta(z_t, I, t)
-------------------

\dot{\tau}(t)(z_1 - z_0)
\right|_2^2
\right]
$$

さらに，source generatorが予測する平均 $\mu_\theta(I)$ が，ターゲット表現 $z_1$ と近い方向を向くように，正規化後のMSEを補助損失として用いる．

$$
\mathcal{L}_{\mathrm{align}}
============================

\mathbb{E}
\left[
\left|
\frac{\mu_\theta(I)}{|\mu_\theta(I)|_2}
---------------------------------------

\frac{z_1}{|z_1|_2}
\right|_2^2
\right]
$$

ここで，正規化は各画素におけるチャネル方向に対して行う．この損失により，source分布の平均がターゲットマスク表現に近い方向へ配置されるように学習される．

また，source分布の分散を学習する場合には，分散が過度に大きくならないように以下の正則化項を用いる．

$$
\mathcal{L}_{\mathrm{var}}
==========================

\frac{1}{2}
\mathbb{E}
\left[
\exp(\log \sigma^2)
-------------------

## \log \sigma^2

1
\right]
$$

固定分散を用いる設定では，この分散正則化項は0として扱う．分散も学習対象とすると速度場の学習が不安定になるため固定分散での実装を基本としている．

最終的な学習損失は以下で表される．

$$
\mathcal{L}
===========

\mathcal{L}*{\mathrm{FM}}
+
\lambda*{\mathrm{var}}\mathcal{L}*{\mathrm{var}}
+
\lambda*{\mathrm{align}}\mathcal{L}_{\mathrm{align}}
$$

推論時には，source generatorから得られる $z_0$ を初期状態とし，学習済み速度場 $v_\theta$ をEuler法，midpoint法，またはRK4法で積分する．最終的に得られた連続表現をカテゴリ確率へ戻し，argmaxによってセマンティックセグメンテーションマスクを得る．


## リポジトリ構成

```text
.
├── configs/
│   └── cityscapes_csfm_128x256.json
├── scripts/
│   ├── train.sh
│   ├── evaluate.sh
│   └── evaluate_prior.sh
├── src/
│   ├── builders.py
│   ├── config.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── evaluate_prior.py
│   ├── flow_matching.py
│   ├── metrics.py
│   ├── models.py
│   ├── train.py
│   └── visualization.py
├── requirements.txt
└── README.md
```

## 環境構築

Python環境を作成し，必要なライブラリをインストールします．

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Weights & Biasesを用いる場合は，追加でインストールしてください．

```bash
pip install wandb
```

`wandb` は任意です．`--use_wandb` を指定しない場合，学習はローカルログのみで実行されます．

## データセット

本コードはCityscapesを対象としています．
Cityscapesデータセットは本リポジトリには含まれていません．各自でダウンロードし，`torchvision.datasets.Cityscapes` が読み込める形式で配置してください．

想定されるディレクトリには，少なくとも以下が含まれます．

```text
leftImg8bit/
gtFine/
```

データセットパスは，configファイルの `data_root`，コマンドライン引数 `--data_root`，またはshell script実行時の環境変数 `DATA_ROOT` で指定できます．

```bash
DATA_ROOT=/path/to/cityscapes bash scripts/train.sh
```

本実装では，Cityscapesの19クラスを `0` から `18` に対応させ，その他のvoid領域をclass `19` として扱います．

## 設定ファイル

標準の実験設定は以下にあります．

```text
configs/cityscapes_csfm_128x256.json
```

このファイルには，以下の設定が含まれます．

* データセット設定
* 学習設定
* optimizer設定
* scheduler設定
* Flow Matching設定
* モデル設定
* source generator設定
* 評価設定
* 任意のwandb設定

学習時には，実際に使用した設定が `<result_dir>/config.json` に保存されます．また，checkpoint内にもconfigが保存されます．評価時には，checkpoint内のconfigを優先してモデル構成を復元します．

## 学習

`scripts/train.sh` の上部にあるパスを必要に応じて編集し，以下を実行します．

```bash
bash scripts/train.sh
```

直接 `src/train.py` を実行する場合は，以下のように指定できます．

```bash
python src/train.py \
  --config configs/cityscapes_csfm_128x256.json \
  --data_root /path/to/cityscapes \
  --result_dir outputs/csfm_cityscapes_128x256 \
  --amp \
  --amp_dtype bf16 \
  --use_checkpoint
```

学習を再開する場合は，以下のように指定します．

```bash
python src/train.py \
  --config configs/cityscapes_csfm_128x256.json \
  --resume outputs/csfm_cityscapes_128x256/model_final.pth \
  --resume_mode full
```

`--resume_mode model_only` を指定すると，optimizerやschedulerの状態を読み込まず，モデル重みのみを読み込みます．

## 評価

通常のセグメンテーション評価は以下で実行します．

```bash
bash scripts/evaluate.sh
```

直接実行する場合は，以下のように指定できます．

```bash
python src/evaluate.py \
  --config configs/cityscapes_csfm_128x256.json \
  --data_root /path/to/cityscapes \
  --ckpt_path outputs/csfm_cityscapes_128x256/model_final.pth \
  --save_dir results/csfm_cityscapes_128x256/evaluation \
  --split val \
  --num_steps 50
```

評価結果は，`metrics.json` と `metrics.txt` に保存されます．また，指定枚数分の生成過程の可視化も保存できます．

## Source Prior評価

source generatorの出力が，ターゲットマスク表現にどの程度近いかを確認するため，source prior評価用のスクリプトを用意しています．

```bash
bash scripts/evaluate_prior.sh
```

デフォルトでは，source generatorの平均 `mu` をdecodeして評価します．
確率的なsource sample `z0` をdecodeする場合は，`--decode_z0` を指定します．

```bash
python src/evaluate_prior.py \
  --config configs/cityscapes_csfm_128x256.json \
  --data_root /path/to/cityscapes \
  --ckpt_path outputs/csfm_cityscapes_128x256/model_final.pth \
  --save_dir results/csfm_cityscapes_128x256/prior \
  --decode_z0 \
  --sample_seed 0
```

## 評価指標

本実装では，以下の指標を計算します．

* pixel accuracy
* mean IoU
* mean class accuracy
* class-wise IoU
* class-wise accuracy
* confusion matrix

## 実験結果

最終的な再現実験の結果を確認後，以下に追加します．

| Dataset        | Resolution | Sampler | Steps | Pixel Accuracy | mIoU | mAcc |
| -------------- | ---------: | ------- | ----: | -------------: | ---: | ---: |
| Cityscapes val |  128 x 256 | TBD     |   TBD |            TBD |  TBD |  TBD |

## 注意事項

以下はGit管理対象に含めていません．

* Cityscapesデータセット
* 学習済みcheckpoint
* 実験出力
* `wandb` ログ
* Python仮想環境

再現実験を行う場合は，各自でCityscapesデータセットと学習済みcheckpointを用意してください．

## ライセンス

ライセンスは後で追加予定です．
