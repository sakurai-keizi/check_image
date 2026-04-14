# check_image

pHash（知覚ハッシュ）を使って画像の類似度を比較・重複検出するツールです。  
`uv run` で依存ライブラリのインストールなしに実行できます。

## スクリプト一覧

| ファイル | 用途 |
|---|---|
| `compare.py` | 2枚の画像を1対1で比較する |
| `extract_features.py` | ディレクトリ内の画像から特徴量を抽出し、類似ペアを検索する |

---

## compare.py — 2枚の画像を比較する

```bash
uv run compare.py <image1> <image2> [--threshold N]
```

### 引数

| 引数 | 説明 | デフォルト |
|---|---|---|
| `image1` | 1枚目の画像ファイルパス | （必須） |
| `image2` | 2枚目の画像ファイルパス | （必須） |
| `--threshold N` | 類似と判定するハミング距離の閾値 | `10` |

### 使用例

```bash
uv run compare.py photo1.jpg photo2.jpg
uv run compare.py photo1.jpg photo2.jpg --threshold 5
```

### 出力例

```
画像1: photo1.jpg
画像2: photo2.jpg
ハミング距離: 3 / 64
類似度スコア: 0.9531
判定 (閾値=10): ほぼ同じ写真
```

---

## extract_features.py — 大量画像の特徴量抽出と類似ペア検索

2つのサブコマンドで構成されています。

### extract — 特徴量を抽出して保存する

```bash
uv run extract_features.py extract <directory> [オプション]
```

| オプション | 説明 | デフォルト |
|---|---|---|
| `directory` | 対象ディレクトリのパス | （必須） |
| `-o, --output FILE` | 特徴量の出力先JSONファイル | `features.json` |
| `-r, --recursive` | サブディレクトリを再帰的に検索する | オフ |
| `-w, --workers N` | 並列プロセス数 | CPUコア数 |

```bash
# 基本
uv run extract_features.py extract /path/to/photos

# サブディレクトリも含める
uv run extract_features.py extract /path/to/photos -r

# 出力先とプロセス数を指定
uv run extract_features.py extract /path/to/photos -o my_features.json -w 4
```

2回目以降の実行では、更新日時（mtime）が変わっていないファイルはスキップされ、差分のみ再計算されます。

### search — 類似画像ペアを検索する

```bash
uv run extract_features.py search [オプション]
```

| オプション | 説明 | デフォルト |
|---|---|---|
| `-f, --features FILE` | 特徴量JSONファイルのパス | `features.json` |
| `-t, --threshold N` | 類似と判定するハミング距離の閾値 | `10` |
| `-o, --output FILE` | 結果をJSONファイルに保存する | （省略可） |

```bash
# 基本
uv run extract_features.py search

# 閾値を厳しくして結果をファイルに保存
uv run extract_features.py search -t 5 -o results.json
```

### 出力例

```
1000 件の特徴量を読み込みました。499500 ペアを比較中...

類似ペアが 3 組見つかりました（閾値: 10）:

[1] ハミング距離: 0  類似度: 1.0000
     /photos/img_001.jpg
     /photos/img_001_copy.jpg
[2] ハミング距離: 3  類似度: 0.9531
     /photos/img_042.jpg
     /photos/img_042_resized.jpg
```

---

## features.json の形式

```json
{
  "/path/to/photo1.jpg": {
    "phash": "f8c0e0c0f0e8c8c0",
    "mtime": 1712345678.0
  }
}
```

---

## pHash について

pHash（知覚ハッシュ）は画像を32×32にリサイズしてDCT変換を行い、64ビットのハッシュ値を生成します。  
2つの画像のハッシュ間のハミング距離が小さいほど類似しています。

### 閾値の目安

| ハミング距離 | 意味 |
|---|---|
| 0 | 完全一致 |
| 1〜5 | ほぼ同じ（圧縮・軽微な色調変化） |
| 6〜10 | 類似（リサイズ・軽微な編集） |
| 11以上 | 異なる写真の可能性が高い |

### 対応している変換

| 変換 | 対応 |
|---|---|
| 解像度違い（4K vs 720p など） | ✅ |
| JPEG圧縮・画質劣化 | ✅ |
| 軽微な明るさ・コントラスト変化 | ✅ |
| 大きなトリミング | △ 閾値次第 |
| 回転・反転 | ❌ |

### 対応画像フォーマット

`.jpg` `.jpeg` `.png` `.webp` `.bmp` `.gif` `.tiff`

---

## 動作要件

- Python 3.11 以上
- [uv](https://github.com/astral-sh/uv)（依存ライブラリは自動インストールされます）
