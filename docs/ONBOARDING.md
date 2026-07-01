# LLMオンボーディングサマリー — WildDet3D

> 新任LLMエージェントが本プロジェクト（このワークスペース）に参加する際の初期資料です。
> 内容は 2026-07-01 時点のワークスペース状態に基づきます。仕様変更時は本書も更新してください。

## 1. プロジェクト概要と目的
- **プロジェクト名称・領域:** WildDet3D — *Scaling Promptable 3D Detection in the Wild*。オープン語彙・単眼3D物体検出（Computer Vision）。Allen Institute for AI (Ai2) ほかの研究成果の公開実装。
- **最終成果物:**
  - アーキテクチャ: SAM3 ViT backbone + LingBot-Depth (DINOv2 ViT-L/14) geometry backend、約1.2Bパラメータ。
  - 提供物: 学習済みモデル重み、推論/評価/学習コード、HuggingFace Space デモ、iPhoneアプリ。
  - **このワークスペースでの直近成果物:** uv で再現可能な推論環境 + 動作する Gradio デモ（バグ修正済み）。
- **ビジネス背景・価値:** テキスト / 点 / ボックスのプロンプトで任意物体を 3D 検出。ロボティクス・AR/VR・アノテーション基盤への応用。
- **現時点の進捗サマリ（2026-07-01）:**
  - ✅ uv 環境（`.venv`, Python 3.11, torch 2.5.1+cu121）構築、推論を E2E 検証済み。
  - ✅ モデル重み取得済み（paper ckpt ~4.73GB + lingbot-depth ~1.28GB）。
  - ✅ Gradio デモ（`demo/huggingface/app.py`）の `@spaces.GPU` フォールバック不具合を修正し、Playwright で E2E 検証済み。
  - ⬜ 評価（`vis4d test`）は未整備 — `vis4d_cuda_ops` のソースビルドが必要だが当環境は CUDA 11.8 のみで cu121 と不整合。

## 2. クリティカルな要求・制約
> 「壊してはいけない」ラインを列挙します。
- **推論に `vis4d_cuda_ops` は不要。** 使用箇所は `wilddet3d/ops/box3d.py` → `wilddet3d/eval/detect3d.py`（AP計算）のみ。推論経路はこの CUDA op を読み込まない構成を維持すること（当環境は CUDA 11.8 のみで cu121 とビルド不整合）。
- **pydantic の上書きを外さない。** gradio 5 は `pydantic>=2`、`vis4d==1.0.0` は宣言上 `pydantic<2`。vis4d は実際には pydantic を import していないため、`pyproject.toml` の `[tool.uv] override-dependencies = ["pydantic>=2"]` で共存させている。
- **`@spaces.GPU` フォールバックの両対応を維持。** `demo/huggingface/app.py` は `@spaces.GPU`（括弧なし）で使用。ローカル用フォールバックは「括弧なし / 括弧あり」両方に対応させること（壊すと Run 時に全 Gradio コールバックが `TypeError`）。
- **大容量ファイルをコミットしない。** `ckpt/` は gitignore。重みは HuggingFace から取得する。生成物（`.venv/`, `outputs/`, `.playwright*/`）もコミットしない。
- **GPU は共有資源。** RTX 4000 Ada（20GB）を他プロジェクトと共有。**他プロセスを kill しない。** OOM 時は GPU が空くまで待機する。
- **ライセンス:** コード / モデルとも SAM License。研究・教育用途に限定。画像データは再配布不可。

## 3. 参照すべき合意済み資料
| 種別 | ファイル/リンク | 概要・用途 |
|------|------------------|------------|
| 全体像 | `README.md` | インストール・推論・評価・学習・結果の一次情報 |
| 推論API | `docs/INFERENCE.md` | `build_model` / `preprocess` / 出力仕様の詳細 |
| 評価 | `docs/EVALUATION.md` | メトリクス定義・データ配置・config一覧 |
| 学習 | `docs/TRAINING.md` | 3ステージ学習手順 |
| データ準備 | `docs/TRAINING_DATA.md` | 各データセットのDL・変換・HDF5化 |
| デモ | `docs/DEMO.md`, `demo/huggingface/README.md` | HuggingFace Gradio デモの構成と起動 |
| 環境定義 | `pyproject.toml`, `uv.lock` | uv 依存定義・ロック（本ワークスペースで追加） |

## 4. タスク境界（任せること / 任せないこと）
### 任せるタスク
- uv 環境の構築・依存管理、推論スクリプトの作成・実行。
- 単純・長時間の機械作業（モデル DL 等）は Sonnet サブエージェントに委譲。
- Playwright によるデモの操作・不具合再現・検証。
- ドキュメント整備、コードの読解・小規模修正。

### 任せないタスク（実行前に要確認）
- 学習（`vis4d fit`）の実行 — 高コスト・多GPU前提。
- 外部への push / 公開・モデル重みの再配布。
- 他プロジェクトの GPU プロセス操作・kill。
- 破壊的な git 操作（force push、履歴書き換え）。

## 5. インタラクション方針
- **回答スタイル:** 日本語、見出し + 箇条書きで簡潔に。要点先出し。
- **回答手順:** 現状確認 → 原因 / 論点 → 提案・実行 → 検証。
- **禁止事項・注意:** 未確定事項を断定しない。外向き操作（push・サーバ公開）は事前確認。他プロジェクトのプロセスに触れない。
- **秘匿情報の扱い:** `HF_TOKEN` 等の認証情報はコミット・ログ出力しない。個人情報を投入しない。

## 6. 試行タスク（オンボーディング演習）
1. `uv sync` 後、`PYTHONPATH=$(pwd)` を設定し README の Inference（text prompt）例をサンプル画像で実行、3D ボックスを出力できることを確認する。
2. `uv sync --group demo` でデモ依存を入れ `demo/huggingface/app.py` を起動、Playwright で `truck.car.wheel` の検出を再現する。
3. `vis4d_cuda_ops` の依存箇所を grep で追跡し、なぜ推論に不要で eval にのみ必要かを説明する。

## 7. 運用ルール・変更管理
- **ドキュメント更新時の記載ルール:** 変更理由を明記し、日付は絶対表記（例: 2026-07-01）。本書は状態が変わったら更新する。
- **TBD の扱い:** 未確定は「TBD」と明示し放置しない。
- **レビュー / 承認フロー:** push 先リポジトリを明示（例: fork `yuki-inaho/WildDet3D`）。`main` への直接反映は事前確認。
- **その他:** コミットメッセージ末尾に Co-Authored-By を付与。大容量・生成物は `.gitignore` で除外。

---

### 付録: 参考情報
- **主要リポジトリ / ディレクトリ:**
  - `wilddet3d/` … コア（model / inference / preprocessing / ops / eval / vis）
  - `demo/huggingface/` … Gradio デモ（`app.py`, `vis3d_glb.py`）
  - `third_party/{sam3, lingbot_depth}` … git submodule。`wilddet3d/__init__.py` が `sys.path` に注入（pip 導入不要）
  - `configs/` … eval / training の config
  - `ckpt/` … モデル重み（gitignore。`wilddet3d.pt` はデモが参照するローカルパス）
- **代表的なコマンド:**
  - `uv sync` / `uv sync --group demo`（デモ依存込み）
  - `export PYTHONPATH=$(pwd); uv run python <script>.py`
  - `uv run python demo/huggingface/app.py`（Gradio, http://localhost:7860）
  - `playwright-cli open http://localhost:7860`（ブラウザ操作・デバッグ）
- **依存ライブラリ:** torch 2.5.1+cu121, torchvision 0.20.1, vis4d==1.0.0, transformers/timm/einops, sam3・mdm(lingbot)（submodule）, gradio 5（demo, pydantic≥2 上書き）。
- **連絡先 / 責任者:** リポジトリオーナー（fork: `yuki-inaho`）。上流: `allenai/WildDet3D`。

> ※本テンプレートは必要に応じて拡張・縮退可。記入済みドキュメントはバージョン管理してください。
