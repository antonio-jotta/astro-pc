from __future__ import annotations

import argparse
import gzip
import shutil
import struct
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import convolve1d


def read_idx_images(path: str | Path) -> np.ndarray:
    """Read an MNIST idx3 image file."""
    path = Path(path)
    with path.open("rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid image file magic number in {path}: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)

    return data.reshape(num_images, rows, cols)


def write_idx_images(path: str | Path, images: np.ndarray) -> None:
    """Write an MNIST idx3 image file."""
    path = Path(path)
    images = np.asarray(images, dtype=np.uint8)

    if images.ndim != 3:
        raise ValueError(f"Expected images with shape [N, H, W], got {images.shape}")

    num_images, rows, cols = images.shape

    with path.open("wb") as f:
        f.write(struct.pack(">IIII", 2051, num_images, rows, cols))
        f.write(images.tobytes())


def gzip_file(path: str | Path) -> None:
    """Create a .gz copy of a file."""
    path = Path(path)
    gz_path = path.with_name(path.name + ".gz")

    with path.open("rb") as f_in, gzip.open(gz_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


def corrupt_gaussian_noise(images: np.ndarray, sigma: float = 18.0) -> np.ndarray:
    noisy = images.astype(np.float32) + np.random.normal(
        loc=0.0,
        scale=sigma,
        size=images.shape,
    ).astype(np.float32)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def corrupt_motion_blur(images: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd integer >= 3.")

    kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
    blurred = convolve1d(
        images.astype(np.float32),
        weights=kernel,
        axis=2,
        mode="nearest",
    )
    return np.clip(blurred, 0, 255).astype(np.uint8)


def corrupt_brightness(images: np.ndarray, factor: float = 1.35) -> np.ndarray:
    bright = images.astype(np.float32) * factor
    return np.clip(bright, 0, 255).astype(np.uint8)


def corrupt_pixelate(images: np.ndarray, downsample_size: int = 14) -> np.ndarray:
    out = np.empty_like(images)

    for i in range(images.shape[0]):
        img = Image.fromarray(images[i], mode="L")
        img_small = img.resize(
            (downsample_size, downsample_size), resample=Image.BILINEAR
        )
        img_pixel = img_small.resize((28, 28), resample=Image.NEAREST)
        out[i] = np.array(img_pixel, dtype=np.uint8)

    return out


def create_corrupted_mnist_dataset(
    source_raw_dir: str | Path,
    output_dataset_dir: str | Path,
    corruption_name: str,
    corruption_fn,
    *,
    overwrite: bool = False,
) -> None:
    """Create a corrupted MNIST dataset with the standard raw-file structure."""
    source_raw_dir = Path(source_raw_dir)
    output_dataset_dir = Path(output_dataset_dir)
    output_raw_dir = output_dataset_dir / "raw"

    if output_dataset_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output dataset already exists: {output_dataset_dir}\n"
                f"Use --overwrite if you want to replace it."
            )
        shutil.rmtree(output_dataset_dir)

    output_raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCreating {corruption_name} dataset at: {output_dataset_dir}")

    train_images = read_idx_images(source_raw_dir / "train-images-idx3-ubyte")
    test_images = read_idx_images(source_raw_dir / "t10k-images-idx3-ubyte")

    print("  Corrupting training images...")
    corrupted_train = corruption_fn(train_images)

    print("  Corrupting test images...")
    corrupted_test = corruption_fn(test_images)

    train_img_out = output_raw_dir / "train-images-idx3-ubyte"
    test_img_out = output_raw_dir / "t10k-images-idx3-ubyte"

    write_idx_images(train_img_out, corrupted_train)
    write_idx_images(test_img_out, corrupted_test)

    gzip_file(train_img_out)
    gzip_file(test_img_out)

    label_files = [
        "train-labels-idx1-ubyte",
        "train-labels-idx1-ubyte.gz",
        "t10k-labels-idx1-ubyte",
        "t10k-labels-idx1-ubyte.gz",
    ]

    for fname in label_files:
        shutil.copy2(source_raw_dir / fname, output_raw_dir / fname)

    print(f"  Done: {corruption_name}")


def main() -> None:
    source_raw_dir_default = "data/MNIST/raw"
    output_root_default = "data"
    overwrite_default = False
    seed_default = 0

    gaussian_sigma = 70.0
    motion_blur_kernel_size = 11
    brightness_factor = 2.60
    pixelate_downsample_size = 3

    parser = argparse.ArgumentParser(
        description="Create corrupted MNIST datasets from an existing MNIST/raw folder."
    )
    parser.add_argument(
        "--source_raw_dir",
        type=str,
        default=source_raw_dir_default,
        help=f"Path to the source MNIST raw directory. Default: {source_raw_dir_default}",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=output_root_default,
        help=f"Root directory where the corrupted datasets will be created. Default: {output_root_default}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=overwrite_default,
        help="Overwrite existing corrupted dataset folders if they already exist.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=seed_default,
        help=f"Random seed for reproducible corruptions. Default: {seed_default}",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    source_raw_dir = Path(args.source_raw_dir)
    output_root = Path(args.output_root)

    if not source_raw_dir.exists():
        raise FileNotFoundError(f"Source raw directory not found: {source_raw_dir}")

    required_files = [
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
    ]
    for fname in required_files:
        if not (source_raw_dir / fname).exists():
            raise FileNotFoundError(
                f"Missing required MNIST raw file: {source_raw_dir / fname}"
            )

    dataset_specs = [
        (
            "MNIST_gaussian_noise",
            "gaussian_noise",
            lambda x: corrupt_gaussian_noise(x, sigma=gaussian_sigma),
        ),
        (
            "MNIST_motion_blur",
            "motion_blur",
            lambda x: corrupt_motion_blur(x, kernel_size=motion_blur_kernel_size),
        ),
        (
            "MNIST_brightness",
            "brightness",
            lambda x: corrupt_brightness(x, factor=brightness_factor),
        ),
        (
            "MNIST_pixelate",
            "pixelate",
            lambda x: corrupt_pixelate(x, downsample_size=pixelate_downsample_size),
        ),
    ]

    print(f"Source MNIST raw directory: {source_raw_dir}")
    print(f"Output root: {output_root}")
    print("Corruption settings:")
    print(f"  Gaussian sigma: {gaussian_sigma}")
    print(f"  Motion blur kernel size: {motion_blur_kernel_size}")
    print(f"  Brightness factor: {brightness_factor}")
    print(f"  Pixelate downsample size: {pixelate_downsample_size}")

    for folder_name, corruption_name, corruption_fn in dataset_specs:
        create_corrupted_mnist_dataset(
            source_raw_dir=source_raw_dir,
            output_dataset_dir=output_root / folder_name,
            corruption_name=corruption_name,
            corruption_fn=corruption_fn,
            overwrite=args.overwrite,
        )

    print("\nAll corrupted MNIST datasets created successfully.")


if __name__ == "__main__":
    main()
