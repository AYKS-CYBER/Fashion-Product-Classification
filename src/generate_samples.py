"""
generate_samples.py - Generates sample fashion images for local testing.
Saves sample PNGs for shirts, shoes, bags, coats, sneakers, and other categories.
"""

import os
import sys
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import CLASS_NAMES, export_sample_images


def main():
    target_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "samples")
    os.makedirs(target_dir, exist_ok=True)
    print(f"Generating sample fashion product images in: {target_dir}")
    saved = export_sample_images(output_dir=target_dir)
    print("\nGenerated samples:")
    for name, path in saved:
        print(f"  - {name:<15}: {os.path.basename(path)}")


if __name__ == "__main__":
    main()
