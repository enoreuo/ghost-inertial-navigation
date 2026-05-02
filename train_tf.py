"""Training Script for TLIO EKF2 Replacement (TensorFlow/Keras)

Trains the ResNet-1D model on IMU data to predict displacement.
Supports two loss functions:
  - huber (default): Robust regression, stable for small datasets
  - nll: Negative log-likelihood with learned uncertainty

Usage:
    python train_tf.py
    python train_tf.py --loss huber --dropout 0.3 --epochs 200
    python train_tf.py --loss nll --epochs 100
"""

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from model_arch_tf import create_tlio_model, create_tlio_model_v2
from model_arch_v25_tflm import create_tlio_model_v25
from load_data_tf import get_data


def nll_loss(y_true, y_pred):
    """Negative log-likelihood loss with learned uncertainty.

    y_pred[:, :3] = displacement prediction
    y_pred[:, 3:] = log_variance (uncertainty)
    y_true = ground truth displacement (3,)

    Loss = 0.5 * (error^2 / variance + log(variance))
    """
    displacement_pred = y_pred[:, :3]
    log_var = y_pred[:, 3:]
    variance = tf.exp(log_var)

    loss = 0.5 * (tf.square(displacement_pred - y_true) / variance + log_var)
    return tf.reduce_mean(loss)


def huber_loss(y_true, y_pred):
    """Huber loss on displacement only (ignores log_variance outputs).

    Robust to outliers, stable for small datasets.
    """
    displacement_pred = y_pred[:, :3]
    return tf.reduce_mean(tf.keras.losses.huber(y_true, displacement_pred))


def displacement_mae(y_true, y_pred):
    """MAE on displacement only (first 3 outputs), in meters."""
    displacement_pred = y_pred[:, :3]
    return tf.reduce_mean(tf.abs(displacement_pred - y_true))


def main():
    parser = argparse.ArgumentParser(description='Train TLIO EKF2 Replacement (TensorFlow)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--num_blocks', type=int, default=2)
    parser.add_argument('--window_size', type=int, default=200)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--dilated_dim', type=int, default=32,
                        help='Dilated conv branch output dimension (v2.5 only)')
    parser.add_argument('--loss', type=str, default='huber', choices=['nll', 'huber'])
    parser.add_argument('--arch', type=str, default='v1', choices=['v1', 'v2', 'v25'],
                        help='v1=ResNet-1D (original), v2=Multi-Scale+FFT+SE (enhanced), v25=v2 with dilated conv (TFLM-compatible)')
    parser.add_argument('--data_dir', type=str, default=None)
    args = parser.parse_args()

    print(f"TensorFlow: {tf.__version__}")
    print(f"GPU: {tf.config.list_physical_devices('GPU')}")
    config_str = (f"Config: arch={args.arch}, epochs={args.epochs}, batch_size={args.batch_size}, "
                  f"lr={args.lr}, hidden={args.hidden_channels}, blocks={args.num_blocks}, "
                  f"dropout={args.dropout}, loss={args.loss}")
    if args.arch == 'v25':
        config_str += f", dilated_dim={args.dilated_dim}"
    print(config_str)
    print()

    # Load data
    (X_train, y_train), (X_test, y_test) = get_data(
        data_dir=args.data_dir,
        window_size=args.window_size
    )
    print(f"\nDataset: {X_train.shape[0]} train, {X_test.shape[0]} test")
    print(f"Input shape: {X_train.shape} -> (samples, {args.window_size} IMU steps, 6 channels)")
    print(f"Output shape: {y_train.shape} -> (samples, 3) = displacement (dx, dy, dz)")
    print()

    # Create model
    if args.arch == 'v2':
        model = create_tlio_model_v2(
            input_channels=6,
            window_size=args.window_size,
            hidden_channels=args.hidden_channels,
            num_blocks=args.num_blocks,
            output_size=3,
            dropout_rate=args.dropout
        )
        save_name = 'tlio_ekf2_v2_tf.keras'
    elif args.arch == 'v25':
        model = create_tlio_model_v25(
            input_channels=6,
            window_size=args.window_size,
            hidden_channels=args.hidden_channels,
            num_blocks=args.num_blocks,
            output_size=3,
            dropout_rate=args.dropout,
            dilated_dim=args.dilated_dim
        )
        save_name = 'tlio_ekf2_v25_tflm.keras'
    else:
        model = create_tlio_model(
            input_channels=6,
            window_size=args.window_size,
            hidden_channels=args.hidden_channels,
            num_blocks=args.num_blocks,
            output_size=3,
            dropout_rate=args.dropout
        )
        save_name = 'tlio_ekf2_tf.keras'
    model.summary()
    print()

    # Select loss function
    loss_fn = huber_loss if args.loss == 'huber' else nll_loss
    loss_name = args.loss.upper()

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=loss_fn,
        metrics=[displacement_mae]
    )

    # Callbacks
    checkpoint = ModelCheckpoint(
        save_name,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-5,
        verbose=1
    )

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[checkpoint, lr_scheduler],
        verbose=1
    )

    # Final evaluation
    best_val_loss = min(history.history['val_loss'])
    best_val_mae = min(history.history['val_displacement_mae'])

    print(f"\nTraining complete!")
    print(f"Best val loss ({loss_name}): {best_val_loss:.6f}")
    print(f"Best val MAE: {best_val_mae:.4f} meters")
    print(f"Model saved to: {save_name}")


if __name__ == '__main__':
    main()
