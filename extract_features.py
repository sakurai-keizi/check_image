# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "Pillow",
#   "imagehash",
#   "tqdm",
# ]
# ///

import argparse
import json
import multiprocessing
import os
import sys
from itertools import combinations
from pathlib import Path

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

    items = [(path, imagehash.hex_to_hash(data["phash"])) for path, data in features.items()]
    total_pairs = len(items) * (len(items) - 1) // 2
    print(f"{len(items)} 件の特徴量を読み込みました。{total_pairs} ペアを比較中...")

    similar_pairs: list[dict] = []
    for (path1, hash1), (path2, hash2) in tqdm(combinations(items, 2), total=total_pairs, unit="ペア"):
        distance = hash1 - hash2
        if distance <= args.threshold:
            similar_pairs.append({
                "image1": path1,
                "image2": path2,
                "hamming_distance": distance,
                "similarity": round(1.0 - distance / 64.0, 4),
            })

    similar_pairs.sort(key=lambda x: x["hamming_distance"])

    if not similar_pairs:
        print(f"\n類似ペアは見つかりませんでした（閾値: {args.threshold}）。")
        return

    print(f"\n類似ペアが {len(similar_pairs)} 組見つかりました（閾値: {args.threshold}）:\n")
    for i, pair in enumerate(similar_pairs, 1):
        print(f"[{i}] ハミング距離: {pair['hamming_distance']}  類似度: {pair['similarity']:.4f}")
        print(f"     {pair['image1']}")
        print(f"     {pair['image2']}")

    if args.output:
        output_path = Path(args.output)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(similar_pairs, f, ensure_ascii=False, indent=2)
        print(f"\n結果を保存しました: {output_path}")


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

    args = parser.parse_args()
    if args.command == "extract":
        cmd_extract(args)
    elif args.command == "search":
        cmd_search(args)


if __name__ == "__main__":
    main()
