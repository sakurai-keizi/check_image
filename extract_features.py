# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "Pillow",
#   "imagehash",
#   "tqdm",
#   "numpy",
# ]
# ///

import argparse
import json
import multiprocessing
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import imagehash
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}


def _compute_phash_worker(image_path: Path) -> tuple[Path, str | None, float]:
    """マルチプロセス用ワーカー。(path, phash, mtime) を返す。"""
    try:
        mtime = image_path.stat().st_mtime
        img = Image.open(image_path)
        return image_path, str(imagehash.phash(img)), mtime
    except Exception as e:
        print(f"  警告: スキップ ({image_path.name}): {e}", file=sys.stderr)
        return image_path, None, 0.0


# ─── extract サブコマンド ────────────────────────────────────────────────────

def cmd_extract(args: argparse.Namespace) -> None:
    dir_path = Path(args.directory).resolve()
    if not dir_path.is_dir():
        print(f"エラー: ディレクトリが見つかりません: {args.directory}", file=sys.stderr)
        sys.exit(1)

    glob = dir_path.rglob("*") if args.recursive else dir_path.glob("*")
    image_paths = sorted(
        p for p in glob if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        print("画像ファイルが見つかりませんでした。")
        return

    print(f"{len(image_paths)} 件の画像が見つかりました。特徴量を計算中...")

    output_path = Path(args.output)
    existing: dict = {}
    if output_path.exists():
        with output_path.open(encoding="utf-8") as f:
            existing = json.load(f)
        print(f"既存データ {len(existing)} 件を読み込みました（差分更新モード）")

    # mtime が変わっていないファイルはスキップ、変わったものだけ再計算対象にする
    to_compute: list[Path] = []
    features: dict = {}
    skipped = 0

    for img_path in image_paths:
        key = str(img_path)
        mtime = img_path.stat().st_mtime
        if key in existing and existing[key]["mtime"] == mtime:
            features[key] = existing[key]
            skipped += 1
        else:
            to_compute.append(img_path)

    if to_compute:
        workers = args.workers or os.cpu_count() or 1
        print(f"  {len(to_compute)} 件を {workers} プロセスで並列処理中...")
        added = updated = 0
        with multiprocessing.Pool(processes=workers) as pool:
            results = pool.imap_unordered(_compute_phash_worker, to_compute)
            for img_path, phash, mtime in tqdm(results, total=len(to_compute), unit="枚"):
                if phash is None:
                    continue
                key = str(img_path)
                is_update = key in existing
                features[key] = {"phash": phash, "mtime": mtime}
                if is_update:
                    updated += 1
                else:
                    added += 1
    else:
        added = updated = 0

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(features, f, ensure_ascii=False, indent=2)

    print(f"\n完了: {output_path}")
    print(f"  新規追加: {added} 件 / 更新: {updated} 件 / スキップ: {skipped} 件")
    print(f"  合計保存: {len(features)} 件")


# ─── search サブコマンド ─────────────────────────────────────────────────────

def _popcount_uint64(x: np.ndarray) -> np.ndarray:
    """numpy uint64 配列の各要素のビット数を返す（ハミング重み）。"""
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


def _group_pairs(similar_pairs: list[dict]) -> list[list[str]]:
    """Union-Find で類似ペアを連結成分（グループ）にまとめる。
    枚数の多いグループ順に返す。
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        if parent.setdefault(x, x) != x:
            parent[x] = find(parent[x])
        return parent[x]

    for pair in similar_pairs:
        root1, root2 = find(pair["image1"]), find(pair["image2"])
        if root1 != root2:
            parent[root1] = root2

    groups: dict[str, list[str]] = {}
    for path in parent:
        groups.setdefault(find(path), []).append(path)

    return sorted(groups.values(), key=len, reverse=True)


def _write_html_report(
    groups: list[list[str]],
    similar_pairs: list[dict],
    output_path: Path,
) -> None:
    """類似グループをHTMLレポートとして出力する。"""
    # ペアの距離をすばやく引けるようにルックアップテーブルを作成
    dist_map: dict[tuple[str, str], int] = {}
    for pair in similar_pairs:
        dist_map[(pair["image1"], pair["image2"])] = pair["hamming_distance"]
        dist_map[(pair["image2"], pair["image1"])] = pair["hamming_distance"]

    total_images = sum(len(g) for g in groups)
    cards = []
    for gi, group in enumerate(groups, 1):
        figures = []
        for path in group:
            src = Path(path).resolve().as_uri()
            name = Path(path).name
            figures.append(f"""
      <figure>
        <img src="{src}" loading="lazy" alt="{name}">
        <figcaption>{name}</figcaption>
      </figure>""")

        cards.append(f"""
  <div class="group">
    <div class="group-header">
      <span class="group-rank">グループ {gi}</span>
      <span class="group-count">{len(group)} 枚</span>
    </div>
    <div class="images">{"".join(figures)}
    </div>
  </div>""")

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>類似画像レポート</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: sans-serif; background: #f4f4f4; color: #333; padding: 24px; }}
    h1 {{ font-size: 1.4rem; margin-bottom: 4px; }}
    .summary {{ color: #666; font-size: 0.9rem; margin-bottom: 24px; }}
    .group {{
      background: #fff; border-radius: 8px; padding: 16px;
      margin-bottom: 16px; box-shadow: 0 1px 4px rgba(0,0,0,.1);
    }}
    .group-header {{
      display: flex; align-items: center; gap: 12px;
      margin-bottom: 12px;
    }}
    .group-rank {{ font-weight: bold; font-size: 1rem; color: #555; }}
    .group-count {{
      font-size: 0.85rem; background: #e8f0fb; color: #1a6fcf;
      padding: 2px 8px; border-radius: 10px;
    }}
    .images {{ display: flex; gap: 12px; flex-wrap: wrap; }}
    figure {{ flex: 1; min-width: 160px; max-width: 240px; }}
    img {{
      width: 100%; max-height: 240px; object-fit: contain;
      background: #eee; border-radius: 4px; display: block;
    }}
    figcaption {{
      font-size: 0.75rem; color: #888; margin-top: 4px;
      word-break: break-all;
    }}
  </style>
</head>
<body>
  <h1>類似画像レポート</h1>
  <p class="summary">{len(groups)} グループ（計 {total_images} 枚）が見つかりました</p>
{"".join(cards)}
</body>
</html>"""
    output_path.write_text(html, encoding="utf-8")


def cmd_search(args: argparse.Namespace) -> None:
    features_path = Path(args.features)
    if not features_path.exists():
        print(f"エラー: 特徴量ファイルが見つかりません: {args.features}", file=sys.stderr)
        print("先に extract サブコマンドで特徴量を生成してください。", file=sys.stderr)
        sys.exit(1)

    with features_path.open(encoding="utf-8") as f:
        features: dict = json.load(f)

    if len(features) < 2:
        print("比較できる画像が2件未満です。")
        return

    paths = list(features.keys())
    hashes = np.array(
        [int(data["phash"], 16) for data in features.values()],
        dtype=np.uint64,
    )
    n = len(paths)
    total_pairs = n * (n - 1) // 2
    print(f"{n} 件の特徴量を読み込みました。{total_pairs:,} ペアを比較中...")

    similar_pairs: list[dict] = []
    for i in tqdm(range(n - 1), unit="枚"):
        # hashes[i] と hashes[i+1:] を一括比較
        xor = hashes[i] ^ hashes[i + 1:]
        distances = _popcount_uint64(xor).astype(np.int32)
        matched = np.where(distances <= args.threshold)[0]
        for offset in matched:
            j = i + 1 + int(offset)
            dist = int(distances[offset])
            similar_pairs.append({
                "image1": paths[i],
                "image2": paths[j],
                "hamming_distance": dist,
                "similarity": round(1.0 - dist / 64.0, 4),
            })

    similar_pairs.sort(key=lambda x: x["hamming_distance"])

    if not similar_pairs:
        print(f"\n類似ペアは見つかりませんでした（閾値: {args.threshold}）。")
        return

    groups = _group_pairs(similar_pairs)
    total_images = sum(len(g) for g in groups)
    print(f"\n{len(groups)} グループ（計 {total_images} 枚）が見つかりました（閾値: {args.threshold}）:\n")
    for gi, group in enumerate(groups, 1):
        print(f"グループ {gi} ({len(group)} 枚)")
        for path in group:
            print(f"  {path}")

    if args.output:
        output_path = Path(args.output)
        output_data = [{"group": gi, "images": group} for gi, group in enumerate(groups, 1)]
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n結果を保存しました: {output_path}")

    if args.html:
        html_path = Path(args.html)
        _write_html_report(groups, similar_pairs, html_path)
        print(f"HTMLレポートを保存しました: {html_path}")


# ─── エントリーポイント ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="画像のpHash特徴量を管理し、類似画像ペアを検索します"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # extract サブコマンド
    p_extract = subparsers.add_parser("extract", help="ディレクトリ内の画像から特徴量を抽出して保存する")
    p_extract.add_argument("directory", help="対象ディレクトリのパス")
    p_extract.add_argument(
        "-o", "--output",
        default="features.json",
        help="出力ファイルパス（デフォルト: features.json）",
    )
    p_extract.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="サブディレクトリも再帰的に検索する",
    )
    p_extract.add_argument(
        "-w", "--workers",
        type=int,
        default=None,
        help="並列プロセス数（デフォルト: CPUコア数）",
    )

    # search サブコマンド
    p_search = subparsers.add_parser("search", help="特徴量ファイルから類似画像ペアを検索する")
    p_search.add_argument(
        "-f", "--features",
        default="features.json",
        help="特徴量ファイルのパス（デフォルト: features.json）",
    )
    p_search.add_argument(
        "-t", "--threshold",
        type=int,
        default=10,
        help="類似と判定するハミング距離の閾値（デフォルト: 10）",
    )
    p_search.add_argument(
        "-o", "--output",
        help="類似ペアの結果をJSONファイルに保存する（省略可）",
    )
    p_search.add_argument(
        "--html",
        help="類似ペアのHTMLレポートを保存する（省略可）",
    )

    args = parser.parse_args()
    if args.command == "extract":
        cmd_extract(args)
    elif args.command == "search":
        cmd_search(args)


if __name__ == "__main__":
    main()
