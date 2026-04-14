# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "Pillow",
#   "imagehash",
# ]
# ///

import argparse
import sys
from pathlib import Path

from PIL import Image
import imagehash


def compare_images(path1: str, path2: str) -> dict:
    """2つの画像の類似度をpHashで比較する。

    Returns:
        dict: {
            "hamming_distance": int,   # 0に近いほど類似（最大64）
            "similarity": float,       # 0.0〜1.0（1.0が完全一致）
            "is_similar": bool,        # 閾値以下なら類似と判定
        }
    """
    img1 = Image.open(path1)
    img2 = Image.open(path2)

    hash1 = imagehash.phash(img1)
    hash2 = imagehash.phash(img2)

    distance = hash1 - hash2
    similarity = 1.0 - distance / 64.0  # pHashは64ビット

    THRESHOLD = 10  # ハミング距離10以下をほぼ同じ写真と判定
    return {
        "hamming_distance": distance,
        "similarity": similarity,
        "is_similar": distance <= THRESHOLD,
    }


def main():
    parser = argparse.ArgumentParser(description="2つの画像の類似度をpHashで比較します")
    parser.add_argument("image1", type=str, help="1枚目の画像ファイルパス")
    parser.add_argument("image2", type=str, help="2枚目の画像ファイルパス")
    parser.add_argument(
        "--threshold",
        type=int,
        default=10,
        help="類似と判定するハミング距離の閾値（デフォルト: 10）",
    )
    args = parser.parse_args()

    for path in [args.image1, args.image2]:
        if not Path(path).exists():
            print(f"エラー: ファイルが見つかりません: {path}", file=sys.stderr)
            sys.exit(1)

    result = compare_images(args.image1, args.image2)
    result["is_similar"] = result["hamming_distance"] <= args.threshold

    print(f"画像1: {args.image1}")
    print(f"画像2: {args.image2}")
    print(f"ハミング距離: {result['hamming_distance']} / 64")
    print(f"類似度スコア: {result['similarity']:.4f}")
    print(f"判定 (閾値={args.threshold}): {'ほぼ同じ写真' if result['is_similar'] else '異なる写真'}")


if __name__ == "__main__":
    main()
