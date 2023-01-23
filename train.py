import argparse
import glob
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc
from tqdm import tqdm

from compdas.data import *
from compdas.model import *


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # High-level options.
    parser.add_argument("--mode", type=str, default="train", help="train")
    parser.add_argument("--model_path", default="model", help="Path where to save/load the trained model.")
    parser.add_argument("--lambda", type=float, default=10.0, dest="lmbda", help="Lambda for rate-distortion tradeoff.")
    parser.add_argument("--num_filters", type=int, default=128, help="Number of filters per layer.")

    parser.add_argument("--data_path", type=str, default="data", help="data path")
    parser.add_argument("--result_path", type=str, default="results", help="result path")
    parser.add_argument("--format", type=str, default="h5", help="data format")
    parser.add_argument("--nx", type=int, default=512, help="path size for height (spatial axis)")
    parser.add_argument("--nt", type=int, default=512, help="path size for width (temporal axis)")

    parser.add_argument(
        "--log_path",
        default="logs",
        help="Path where to log training metrics for TensorBoard and back up " "intermediate model checkpoints.",
    )
    parser.add_argument("--batch", type=int, default=8, help="Batch size for training and validation.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
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
        default=64,
        help="Maximum number of batches to use for validation. If -1, use one "
        "patch from each image in the training set.",
    )
    parser.add_argument(
        "--precision_policy", type=str, default=None, help="Policy for `tf.keras.mixed_precision` training."
    )
    parser.add_argument(
        "--check_numerics", action="store_true", help="Enable TF support for catching NaN and Inf in tensors."
    )

    args = parser.parse_args()
    return args


def gen_dataset(args):

    files = sorted(list(glob.glob(args.data_path + "/*." + args.format)))
    if not files:
        raise RuntimeError(f"No training images found with glob " f"'{args.data_path}'.")
    for filename in files:
        data = read_hdf5(args, filename)
        data = torch.permute(data, (1, 2, 0))
        h, w, nc = data.shape  # nx, nt, nc
        if args.mode == "train":
            ih = np.random.randint(0, h - args.nx, h // args.nx * 50)
            iw = np.random.randint(0, w - args.nt, w // args.nt * 50)
        else:
            ih = np.arange(0, h - args.nx, args.nx)
            iw = np.arange(0, w - args.nt, args.nt)
        for i in ih:
            for j in iw:
                patch = tf.convert_to_tensor(data[i : i + args.nx, j : j + args.nt, :], dtype=tf.float32)
                yield tf.cast(patch, tf.keras.mixed_precision.global_policy().compute_dtype)


def data_loader(args, split="train"):
    with tf.device("/cpu:0"):
        dataset = tf.data.Dataset.from_generator(
            partial(gen_dataset, args=args), output_signature=tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32)
        )
        if split == "train":
            dataset = dataset.repeat()
        if args.batch > 0:
            dataset = dataset.batch(args.batch)
    return dataset


def plot_data(args, filename, data_true, data_hat):

    if isinstance(data_true, tf.Tensor):
        data_true = data_true.numpy()
    if isinstance(data_hat, tf.Tensor):
        data_hat = data_hat.numpy()

    plt.clf()
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(6 * args.nt / args.nx, 4),
        sharex=True,
        sharey=False,
        squeeze=False,
        gridspec_kw={"wspace": 0, "hspace": 0},
    )
    axes[0, 0].imshow(data_true[:, :, 0], vmin=-1.5, vmax=1.5, cmap="seismic")
    axes[1, 0].imshow(data_hat[:, :, 0], vmin=-1.5, vmax=1.5, cmap="seismic")
    fig.subplots_adjust(wspace=None, hspace=None)
    fig.savefig(filename, dpi=300, bbox_inches="tight")


def test(args):
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    if not os.path.exists(args.result_path + "/figures"):
        os.makedirs(args.result_path + "/figures")

    print(f"Loading model from: {args.model_path}")
    try:
        model = tf.keras.models.load_model(args.model_path)
    except:
        model = BLS2017Model(args.lmbda, args.num_filters)
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        status = checkpoint.restore(tf.train.latest_checkpoint(args.model_path))
        print(f"Restored from checkpoint {args.model_path}: {status}")
    test_dataset = data_loader(args, "test")

    for i, x in enumerate(test_dataset):

        tensors = model.compress(x)

        # # Write a binary file with the shape information and the compressed string.
        packed = tfc.PackedTensors()
        packed.pack(tensors)
        # with open(args.output_file, "wb") as f:
        #     f.write(packed.string)

        x_hat = model.decompress(*tensors)

        plot_data(args, args.result_path + f"/figures/{i:04d}.png", x, x_hat)

        # Cast to float in order to compute metrics.
        x = tf.cast(x, tf.float32)
        x_hat = tf.cast(x_hat, tf.float32)
        mse = tf.reduce_mean(tf.math.squared_difference(x, x_hat))
        psnr = tf.squeeze(tf.image.psnr(x, x_hat, 3))
        msssim = tf.squeeze(tf.image.ssim_multiscale(x, x_hat, 3))
        msssim_db = -10.0 * tf.math.log(1 - msssim) / tf.math.log(10.0)

        # The actual bits per pixel including entropy coding overhead.
        num_pixels = tf.reduce_prod(tf.shape(x)[:-1])
        bpp = len(packed.string) * 8 / num_pixels

        print("----------------------------------------")
        print(f"Mean squared error: {mse:0.4f}")
        print(f"Bits per pixel: {bpp:0.4f}")
        print(f"PSNR (dB): {psnr:0.2f}")
        print(f"Multiscale SSIM: {msssim:0.4f}")
        print(f"Multiscale SSIM (dB): {msssim_db:0.2f}")


def train(args):
    """Instantiates and trains the model."""

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if not os.path.exists(args.model_path + "/checkpoint"):
        os.makedirs(args.model_path + "/checkpoint")

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

    class CustomCallback(tf.keras.callbacks.Callback):
        pass

    model.fit(
        train_dataset.prefetch(8),
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        validation_data=validation_dataset.cache(),
        validation_freq=1,
        callbacks=[
            # CustomCallback(),
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.TensorBoard(log_dir=args.log_path, histogram_freq=1, update_freq="epoch"),
            tf.keras.callbacks.BackupAndRestore(args.log_path),
            # tf.keras.callbacks.ModelCheckpoint(
            #     args.model_path + "/checkpoint/variables",
            #     monitor="val_mse",
            #     save_best_only=False,
            #     save_weights_only=False,
            #     verbose=1,
            # ),
        ],
        verbose=1,
    )

    print(f"Training complete. Saving model to {args.model_path}.")
    model.save(args.model_path)


def main(args):
    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
    else:
        raise ValueError(f"Unknown command {args.command}.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
