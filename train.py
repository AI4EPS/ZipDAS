import argparse
import glob
import sys
from absl import app
from absl.flags import argparse_flags
import tensorflow as tf
import tensorflow_compression as tfc
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from compdas.model import *
from compdas.data import *


def parse_args():
    """Parses command line arguments."""
    parser = argparse_flags.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # High-level options.
    parser.add_argument("--model_path", default="model", help="Path where to save/load the trained model.")
    parser.add_argument("--lambda", type=float, default=100, dest="lmbda", help="Lambda for rate-distortion tradeoff.")
    parser.add_argument(
        "--train_glob",
        type=str,
        default=None,
        help="Glob pattern identifying custom training data. This pattern must "
        "expand to a list of RGB images in PNG format. If unspecified, the "
        "CLIC dataset from TensorFlow Datasets is used.",
    )
    parser.add_argument("--num_filters", type=int, default=128, help="Number of filters per layer.")
    parser.add_argument(
        "--log_path",
        default="/tmp/train_bls2017",
        help="Path where to log training metrics for TensorBoard and back up " "intermediate model checkpoints.",
    )
    parser.add_argument("--batchsize", type=int, default=8, help="Batch size for training and validation.")
    parser.add_argument("--patchsize", type=int, default=256, help="Size of image patches for training and validation.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Train up to this number of epochs. (One epoch is here defined as "
        "the number of steps given by --steps_per_epoch, not iterations "
        "over the full training dataset.)",
    )
    parser.add_argument(
        "--steps_per_epoch", type=int, default=1000, help="Perform validation and produce logs after this many batches."
    )
    parser.add_argument(
        "--max_validation_steps",
        type=int,
        default=256,
        help="Maximum number of batches to use for validation. If -1, use one "
        "patch from each image in the training set.",
    )
    parser.add_argument(
        "--preprocess_threads",
        type=int,
        default=16,
        help="Number of CPU threads to use for parallel decoding of training " "images.",
    )
    parser.add_argument(
        "--precision_policy", type=str, default=None, help="Policy for `tf.keras.mixed_precision` training."
    )
    parser.add_argument(
        "--check_numerics", action="store_true", help="Enable TF support for catching NaN and Inf in tensors."
    )

    parser.add_argument("--data_path", type=str, default="data", help="data path")
    parser.add_argument("--result_path", type=str, default="compressed", help="result path")
    parser.add_argument("--format", type=str, default="h5", help="data format")
    parser.add_argument("--nx", type=int, default=256, help="number of pixels in x direction")
    parser.add_argument("--nt", type=int, default=256, help="number of pixels in t direction")
    args = parser.parse_args()
    return args


def write_data(filename, image):
    """Saves an image to a PNG file."""

    if isinstance(image, tf.Tensor):
        image = image.numpy()
    plt.figure()
    plt.imshow(image[:, :, 0].T, vmin=-1.5, vmax=1.5, cmap="seismic")
    plt.colorbar()
    plt.savefig(filename)


def check_image_size(image, patchsize):
    shape = tf.shape(image)
    return shape[0] >= patchsize and shape[1] >= patchsize and shape[-1] == 3


def crop_image(image, patchsize):
    image = tf.image.random_crop(image, (patchsize, patchsize, 1))
    return tf.cast(image, tf.keras.mixed_precision.global_policy().compute_dtype)


def gen_dataset(args):
    files = sorted(list(glob(args.data_path + "/*." + args.format)))
    if not files:
        raise RuntimeError(f"No training images found with glob " f"'{args.train_glob}'.")
    for filename in files:
        data = read_hdf5(filename)
        data = torch.permute(data, (1, 2, 0))
        h, w, nc = data.shape  # nx, nt, nc
        for i in np.random.randint(0, h - args.nx, h // args.nx):
            for j in np.random.randint(0, w - args.nt, w // args.nt):
                patch = tf.convert_to_tensor(data[i : i + args.nx, j : j + args.nt, :], dtype=tf.float32)
                yield tf.cast(patch, tf.keras.mixed_precision.global_policy().compute_dtype)


def data_loader(args, split="train"):
    """Creates input data pipeline from custom PNG images."""
    with tf.device("/cpu:0"):
        dataset = tf.data.Dataset.from_generator(
            partial(gen_dataset, args=args), output_signature=tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32)
        )
        if split == "train":
            dataset = dataset.repeat()
        dataset = dataset.batch(args.batchsize)
    return dataset


def compress(args):
    """Compresses an image."""
    # Load model and use it to compress the image.
    model = tf.keras.models.load_model(args.model_path)
    x = read_data(args.input_file)
    write_data(args.input_file + ".png", x)
    tensors = model.compress(x)

    # Write a binary file with the shape information and the compressed string.
    packed = tfc.PackedTensors()
    packed.pack(tensors)
    with open(args.output_file, "wb") as f:
        f.write(packed.string)

    # If requested, decompress the image and measure performance.
    if args.verbose:
        x_hat = model.decompress(*tensors)
        write_data(args.input_file + "_verbose.png", x_hat)

        # Cast to float in order to compute metrics.
        x = tf.cast(x, tf.float32)
        x_hat = tf.cast(x_hat, tf.float32)
        mse = tf.reduce_mean(tf.math.squared_difference(x, x_hat))
        psnr = tf.squeeze(tf.image.psnr(x, x_hat, 255))
        msssim = tf.squeeze(tf.image.ssim_multiscale(x, x_hat, 255))
        msssim_db = -10.0 * tf.math.log(1 - msssim) / tf.math.log(10.0)

        # The actual bits per pixel including entropy coding overhead.
        num_pixels = tf.reduce_prod(tf.shape(x)[:-1])
        bpp = len(packed.string) * 8 / num_pixels

        print(f"Mean squared error: {mse:0.4f}")
        print(f"PSNR (dB): {psnr:0.2f}")
        print(f"Multiscale SSIM: {msssim:0.4f}")
        print(f"Multiscale SSIM (dB): {msssim_db:0.2f}")
        print(f"Bits per pixel: {bpp:0.4f}")


def main(args):
    """Instantiates and trains the model."""
    if args.precision_policy:
        tf.keras.mixed_precision.set_global_policy(args.precision_policy)
    if args.check_numerics:
        tf.debugging.enable_check_numerics()

    model = BLS2017Model(args.lmbda, args.num_filters)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    )

    train_dataset = data_loader(args, "train")
    validation_dataset = data_loader(args, "validation")
    validation_dataset = validation_dataset.take(args.max_validation_steps)

    model.fit(
        train_dataset.prefetch(8),
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        validation_data=validation_dataset.cache(),
        validation_freq=1,
        callbacks=[
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.TensorBoard(log_dir=args.log_path, histogram_freq=1, update_freq="epoch"),
            tf.keras.callbacks.BackupAndRestore(args.log_path),
        ],
        # verbose=int(args.verbose),
        verbose=1,
    )
    model.save(args.model_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
    # app.run(main, flags_parser=args)
