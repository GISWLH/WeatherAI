<div align="center">

<img src="docs/assets/banner.png" alt="WeatherAI：査読済みの気象・気候 AI モデル 20 種を、読みやすい 1 つの PyTorch ライブラリに" width="100%">

[English](README.md) · [简体中文](README_zh.md) · **日本語**

[![License: MIT](https://img.shields.io/badge/License-MIT-0EA5E9.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-1D4ED8.svg)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-native-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![JAX](https://img.shields.io/badge/JAX-supported-A855F7.svg)](docs/models/neuralgcm.md)
[![Models](https://img.shields.io/badge/models-20-14B8A6.svg)](#model-zoo)
[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97%20Space-GPU%20tests-FFD21E.svg)](https://huggingface.co/spaces/LonghaoWang/weatherai-graphcast-smoke)
[![HF Hub](https://img.shields.io/badge/%F0%9F%A4%97%20Hub-model%20card-FFD21E.svg)](https://huggingface.co/LonghaoWang/WeatherAI)
[![Stars](https://img.shields.io/github/stars/GISWLH/WeatherAI?style=social)](https://github.com/GISWLH/WeatherAI/stargazers)

</div>

**WeatherAI** は、*Nature*・*Science* とその姉妹誌に掲載された気象・気候 AI モデルを、読みやすい PyTorch の `nn.Module` として再実装し、すべての移植を著者の公式コードと数値的に照合したライブラリです。公式の重みを読み込んで原版と比較したり、ファインチューニングしたり、ノート PC でも動く Lite 構成から始めたりできます。

## ✨ 特長

- 🧩 **20 モデルを 1 つのライブラリで** —— Pangu、GraphCast、Aurora、GenCast、NeuralGCM、FuXi-S2S、CAS-Canglong などを、同じ `import` 形式・テンソル入出力で扱えます。モデルごとに環境を作り分ける必要はありません。
- 🔀 **マルチフレームワーク** —— 全モデルをネイティブ PyTorch で提供しつつ、重要な箇所では公式 **JAX** コードを保持（NeuralGCM の学習と降水モデル）し、GraphCast・GenCast・WeatherNext の数値リファレンスとしても利用します。**ONNX** のみで公開されたモデル（FuXi-ENS、FuXi-S2S）は PyTorch モジュールとして復元済みです。CPU と **CUDA** GPU の両方で動作します。
- ✅ **書き直しではなく「検証済み」** —— 14 モデルで公式チェックポイントを strict に読み込め、出力は公式実装と一致します（多くはビット単位で一致）。すべての主張に検証スクリプトと JSON の結果があり、GPU 動作は Hugging Face ZeroGPU Space 上の NVIDIA Blackwell GPU で確認しています。
- 🪶 **フルサイズと Lite** —— 本格的な実験には論文と同じ規模のネットワークを、手軽な試行には 15 モデルで用意した Lite 構成（同じコード・小さな格子）を。ノート PC の CPU や小容量 GPU 1 枚でも学習できます。

## 🚀 クイックスタート

```bash
pip install git+https://github.com/GISWLH/WeatherAI.git      # または: git clone ... && pip install -e ".[dev]"
```

```python
import torch
from weatherai.models import Canglong_lite, GraphCast_lite, Aurora_small

sst  = Canglong_lite()(torch.randn(1, 8, 8, 32, 64))    # 3 か月先までの SST 予測 -> (1, 1, 3, 32, 64)
step = GraphCast_lite()(torch.randn(1, 4, 32, 64))      # 小さな正二十面体メッシュで 6 時間先を 1 ステップ予測
aurora = Aurora_small(pretrained=True)                  # 公式の重み（約 450 MB）をネイティブ PyTorch に strict 読み込み
```

<a id="model-zoo"></a>
## 🗺️ モデル一覧

🔓 公式の重みを読み込み可 · 🎯 公式コードと数値一致 · ⚡ CUDA GPU で動作確認済み · 🪶 Lite 構成あり · 🆕 新規追加。各 **docs** リンクは 1 ページの概要です：使い方、検証した内容、検証していない内容（英語）。

### 🌍 全球中期予報

| Model | Import | Notes |
|-------|--------|-------|
| **Pangu-Weather** | `from weatherai.models import Pangu, Pangu_lite` | 3 次元 Earth-specific Transformer · *Nature* 2023 · [論文](https://doi.org/10.1038/s41586-023-06185-3) · [docs](docs/models/pangu.md) · 🪶 |
| **FuXi** | `from weatherai.models import FuXi` | キューブ埋め込み + Swin-V2 U-Transformer · *npj Clim. Atmos. Sci.* 2023 · [論文](https://doi.org/10.1038/s41612-023-00512-1) · [docs](docs/models/fuxi.md) |
| **FengWu** | `from weatherai.models import FengWu, FengWu_lite` | マルチモーダルな符号化–融合–復号 · *Commun. Earth Environ.* 2025 · [論文](https://doi.org/10.1038/s43247-025-02502-y) · [docs](docs/models/fengwu.md) · 🪶 |
| **GraphCast** | `from weatherai.models import GraphCast, GraphCast_lite` | 格子 ↔ 正二十面体メッシュの GNN、JAX・PhysicsNeMo と段階ごとに一致 · *Science* 2023 · [論文](https://doi.org/10.1126/science.adi2336) · [docs](docs/models/graphcast.md) · 🎯 ⚡ 🪶 |
| **Aurora** | `from weatherai.models import Aurora, Aurora_lite, Aurora_small` | Perceiver + Swin-3D の基盤モデル · *Nature* 2025 · [論文](https://doi.org/10.1038/s41586-025-09005-y) · [docs](docs/models/aurora.md) · 🔓 🎯 ⚡ 🪶 |
| **NeuralGCM** | `from weatherai.models.neuralgcm import NeuralGCMLearnedComponents, train` | 微分可能な力学コア + 学習型物理、JAX で学習 · *Nature* 2024 · [論文](https://doi.org/10.1038/s41586-024-07744-y) · [docs](docs/models/neuralgcm.md) · 🔓 🎯 ⚡ 🪶 |
| **Aardvark Weather** | `from weatherai.models.aardvark import AardvarkE2E, load_official_encoder` | 生の観測データからのエンドツーエンド予報 · *Nature* 2025 · [論文](https://doi.org/10.1038/s41586-025-08897-0) · [docs](docs/models/aardvark.md) · 🔓 🎯 ⚡ 🪶 |

### 🎲 アンサンブル・生成モデル

| Model | Import | Notes |
|-------|--------|-------|
| **GenCast** | `from weatherai.models.gencast import GenCast_lite, GenCastDenoiser_from_official` | 拡散モデルによるアンサンブル：ネイティブのデノイザー + DPM-Solver++ サンプラー · *Nature* 2025 · [論文](https://doi.org/10.1038/s41586-024-08252-9) · [docs](docs/models/gencast.md) · 🔓 🎯 ⚡ 🪶 |
| **FuXi-ENS** | `from weatherai.models.fuxi_ens import FuXiENS, FuXiENS_lite, load_official` | 25 億パラメータの潜在ノイズ型アンサンブル、ONNX から復元 · *Sci. Adv.* 2025 · [論文](https://doi.org/10.1126/sciadv.adu2854) · [docs](docs/models/fuxi_ens.md) · 🔓 🎯 ⚡ 🪶 |
| **ArchesWeather / Gen** | `from weatherai.models.arches import ArchesWeather_lite, ArchesWeatherGen_lite, convert` | 3 次元アテンション予報器 + フローマッチング生成器 · *Sci. Adv.* 2026 · [論文](https://doi.org/10.1126/sciadv.adx2372) · [docs](docs/models/arches.md) · 🔓 🎯 ⚡ 🪶 |
| **NeuralGCM precipitation** | `from weatherai.models.neuralgcm import precip` | 確率的降水チェックポイント + 学習可能な損失（JAX） · *Sci. Adv.* 2026 · [論文](https://doi.org/10.1126/sciadv.adv6891) · [docs](docs/models/neuralgcm_precip.md) · 🔓 ⚡ |

### 🌊 季節内〜季節予測・気候・海洋

| Model | Import | Notes |
|-------|--------|-------|
| **CAS-Canglong** 🆕 | `from weatherai.models.canglong import Canglong, Canglong_lite, load_official` | 3 次元 Earth-specific Transformer による 1〜3 か月先の全球海面水温予測 · *Adv. Sci.* · [論文](https://doi.org/10.1002/advs.78271) · [docs](docs/models/canglong.md) · 🔓 🎯 ⚡ 🪶 |
| **FuXi-S2S** | `from weatherai.models.fuxi_s2s.convert import load_official` | 季節内予測 Swin-V2 アンサンブル（10.4 億パラメータ）、ONNX から復元 · *Nat. Commun.* 2024 · [論文](https://doi.org/10.1038/s41467-024-50714-1) · [docs](docs/models/fuxi_s2s.md) · 🔓 🎯 ⚡ 🪶 |
| **UniCM** | `from weatherai.models.unicm import UniCM, UniCMConfig` | 結合気候モード Transformer（ENSO など） · *Nat. Mach. Intell.* 2026 · [論文](https://doi.org/10.1038/s42256-026-01245-5) · [docs](docs/models/unicm.md) · 🎯 ⚡ 🪶 |
| **ORCA-DL** | `from weatherai.models.orca_dl import ORCADL, load_official` | 全球 3 次元海洋予測（月単位ステップ） · *Sci. Adv.* 2025 · [論文](https://doi.org/10.1126/sciadv.adu2488) · [docs](docs/models/orca_dl.md) · 🔓 🎯 ⚡ |
| **ACE2-ERA5** | `from weatherai.models.ace2 import ACE2, load_official` | 保存則を組み込んだ球面フーリエニューラル演算子の気候エミュレータ · *npj Clim. Atmos. Sci.* 2025 · [論文](https://doi.org/10.1038/s41612-025-01090-0) · [docs](docs/models/ace2.md) · 🔓 🎯 ⚡ |
| **GenFocal** | `from weatherai.models.genfocal import GenFocalNet, reflow_loss, sample_flow` | 生成型バイアス補正フロー（lite コア） · *Nat. Mach. Intell.* 2026 · [論文](https://doi.org/10.1038/s42256-026-01308-7) · [docs](docs/models/genfocal.md) · 🎯 ⚡ 🪶 |

### 🌀 領域予報・熱帯低気圧

| Model | Import | Notes |
|-------|--------|-------|
| **StormCast** | `from weatherai.models.stormcast import StormCast, StormCast_lite, load_official` | 3 km 領域（HRRR）回帰 U-Net + EDM 拡散 · *Sci. Adv.* 2026 · [論文](https://doi.org/10.1126/sciadv.adv0423) · [docs](docs/models/stormcast.md) · 🔓 🎯 ⚡ 🪶 |
| **TropiCycloneNet** | `from weatherai.models.tropicyclonenet import TCNM, load_official` | マルチモーダルな熱帯低気圧の進路 + 強度予測 · *Nat. Commun.* 2025 · [論文](https://doi.org/10.1038/s41467-025-61087-4) · [docs](docs/models/tropicyclonenet.md) · 🔓 🎯 ⚡ |
| **WeatherNext Cyclones** | `from weatherai.models.weathernext_cyclones import WeatherNextCyclonesNet_from_official, WeatherNextCyclonesNative_lite` | 熱帯低気圧向け関数生成ネットワーク（Mini） · *Nature* 2026 · [論文](https://doi.org/10.1038/s41586-026-10953-2) · [docs](docs/models/weathernext_cyclones.md) · 🔓 🎯 ⚡ 🪶 |

さらに詳しく：[全モデルのページ](docs/models/README.md) · [検証ログ](docs/model_status.md) · [ロードマップと追加予定のモデル](docs/roadmap.md)

## 🔬 「検証済み」の意味

各移植について、未改変の公式コード（PyTorch、JAX、onnxruntime のいずれか）とネイティブモジュールに同じ重みと入力を与え、出力を比較しています。スクリプトは [`scripts/`](scripts)、数値結果は [`docs/results/`](docs/results) にあります。GPU では Hugging Face Space 上で [`scripts/gpu_smoke.py`](scripts/gpu_smoke.py) を実行しています。これらは実装の正しさの確認であり、予報精度の主張ではありません。各モデルのページには、検証して**いない**項目も明記しています。

## 🤗 Hugging Face

- **Space** —— [LonghaoWang/weatherai-graphcast-smoke](https://huggingface.co/spaces/LonghaoWang/weatherai-graphcast-smoke)：全モデルの ZeroGPU スモークテスト。
- **Hub** —— [LonghaoWang/WeatherAI](https://huggingface.co/LonghaoWang/WeatherAI)：モデルカードと各モデルのページ。[`.github/workflows/sync_hf.yml`](.github/workflows/sync_hf.yml) が本リポジトリから自動同期します。
- 公式の重みは各著者のもとにあり、それぞれのライセンス（Apache-2.0、MIT、CC-BY、CC-BY-NC(-ND) など）に従います。WeatherAI は必要時にダウンロードするだけで、再配布はしません。

## 🧪 テスト

```bash
pip install -e ".[dev]"
pytest tests/models -q          # 単体テスト。数値リファレンスは tests/models/*/data に保存
```

## 📖 引用

利用したモデルの原論文（上表のリンク）を引用してください。WeatherAI が役立った場合は、本リポジトリ（[`CITATION.cff`](CITATION.cff)）も引用していただけると幸いです。

```bibtex
@software{wang_weatherai,
  author = {Wang, Longhao},
  title  = {{WeatherAI}: a PyTorch model zoo of AI weather and climate models},
  url    = {https://github.com/GISWLH/WeatherAI},
  year   = {2026}
}
```

## 🤝 コントリビュート

Issue と Pull Request を歓迎します。新しいモデルは、ネイティブモジュール、Lite 構成、公式コードとの数値照合スクリプト、単体テスト、`docs/models/` のページが揃った時点で受け付けます。WeatherAI が役に立ったら、⭐ を付けていただけるとより多くの人に届きます。

## ライセンス

[MIT](LICENSE) —— Copyright (c) 2026 Longhao Wang (GISWLH)。一部のコードは [WeatherLearn](https://github.com/lizhuoq/WeatherLearn)（MIT）を改変したものです（[NOTICE](NOTICE) 参照）。モデルの重みは各著者のライセンスに従います。本リポジトリは研究・教育用のコードです。予報精度はデータ・学習・重みに依存し、現業の予報システムを置き換えるものではありません。
