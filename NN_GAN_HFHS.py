### Regression Neural Network
### By Aliye Hashemi
### Henry Ford Health System --- Hermelin Brain Tumor Center

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_xy(excel_path: str | Path, x_sheet: str, y_sheet: str):
    df_x = pd.read_excel(excel_path, sheet_name=x_sheet, header=None).dropna(how="all")
    df_y = pd.read_excel(excel_path, sheet_name=y_sheet, header=None).dropna(how="all")

    x = df_x.to_numpy(dtype=float)
    y = df_y.to_numpy(dtype=float).reshape(-1)
    return x, y


def build_mlp(input_dim: int, hidden, activation: str, l2_strength: float, lr: float, seed: int) -> tf.keras.Model:
    tf.random.set_seed(seed)
    np.random.seed(seed)

    reg = tf.keras.regularizers.l2(l2_strength)

    m = tf.keras.Sequential(name="mlp_regressor")
    m.add(tf.keras.layers.Input(shape=(input_dim,), name="x"))

    for idx, width in enumerate(hidden, start=1):
        m.add(
            tf.keras.layers.Dense(
                width,
                activation=activation,
                kernel_regularizer=reg,
                name=f"h{idx}",
            )
        )

    m.add(
        tf.keras.layers.Dense(
            1,
            activation="linear",
            kernel_regularizer=reg,
            name="yhat",
        )
    )

    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return m


def metrics(y: np.ndarray, yhat: np.ndarray):
    mse = mean_squared_error(y, yhat)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y, yhat)

    # correlations
    pear = pearsonr(y, yhat)[0]
    spear = spearmanr(y, yhat)[0]
    return mse, rmse, mae, pear, spear


def save_model_summary(model: tf.keras.Model, out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        model.summary(print_fn=lambda s: f.write(s + "\n"))


def plot_loss(history, out_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path, dpi=200)
    plt.show()


def plot_true_vs_pred(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)

    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    plt.plot([lo, hi], [lo, hi], "k--")

    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("Test: True vs Predicted")
    plt.grid(True)
    plt.savefig(out_path, dpi=200)
    plt.show()


def plot_error_hist(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    err = y_true - y_pred
    plt.figure(figsize=(7, 5))
    plt.hist(err, bins=20)
    plt.xlabel("Error")
    plt.ylabel("Count")
    plt.title("Test Prediction Error Distribution")
    plt.grid(True)
    plt.savefig(out_path, dpi=200)
    plt.show()


def main():
    # io
    out_dir = ensure_dir(r"C:\nn")
    train_path = r"C:\NN_15.xlsx"
    test_path = r"C:\NN_3.xlsx"
    x_sheet, y_sheet = "Input", "Labels"

    # model/training
    hidden = [32, 16, 8, 4, 2]
    activation = "relu"
    l2_strength = 1e-4
    lr = 5e-4
    epochs = 200
    batch_size = 16
    seed = 42

    x_train, y_train = load_xy(train_path, x_sheet, y_sheet)
    x_test, y_test = load_xy(test_path, x_sheet, y_sheet)

    print(f"Train: X={x_train.shape} y={y_train.shape}")
    print(f"Test : X={x_test.shape}  y={y_test.shape}")

    scaler = StandardScaler()
    xtr = scaler.fit_transform(x_train)
    xte = scaler.transform(x_test)

    model = build_mlp(
        input_dim=xtr.shape[1],
        hidden=hidden,
        activation=activation,
        l2_strength=l2_strength,
        lr=lr,
        seed=seed,
    )
    save_model_summary(model, out_dir / "model_summary.txt")

    hist = model.fit(
        xtr,
        y_train,
        validation_data=(xte, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    yhat_train = model.predict(xtr, verbose=0).ravel()
    yhat_test = model.predict(xte, verbose=0).ravel()

    mse_tr, rmse_tr, mae_tr, pear_tr, spear_tr = metrics(y_train, yhat_train)
    mse_te, rmse_te, mae_te, pear_te, spear_te = metrics(y_test, yhat_test)

    # baseline: mean predictor from training targets
    baseline_pred = np.full_like(y_test, y_train.mean(), dtype=float)
    rmse_baseline = float(np.sqrt(mean_squared_error(y_test, baseline_pred)))

    # normalized rmse as % of test target range (guard against zero range)
    y_range = float(y_test.max() - y_test.min())
    nrmse_pct = float((rmse_te / y_range) * 100) if y_range != 0 else np.nan

    print("\n=== TRAIN ===")
    print(f"MSE: {mse_tr:.6g}")
    print(f"RMSE: {rmse_tr:.6g}")
    print(f"MAE: {mae_tr:.6g}")
    print(f"Pearson r: {pear_tr:.6g}")
    print(f"Spearman ρ: {spear_tr:.6g}")

    print("\n=== TEST ===")
    print(f"MSE: {mse_te:.6g}")
    print(f"RMSE: {rmse_te:.6g}")
    print(f"MAE: {mae_te:.6g}")
    print(f"Pearson r: {pear_te:.6g}")
    print(f"Spearman ρ: {spear_te:.6g}")
    print(f"Baseline RMSE: {rmse_baseline:.6g}")
    print(f"Normalized RMSE (% of range): {nrmse_pct:.6g}")

    if rmse_te > rmse_baseline:
        print("Warning: model RMSE is worse than baseline mean predictor.")

    metrics_df = pd.DataFrame(
        {
            "dataset": ["train", "test"],
            "MSE": [mse_tr, mse_te],
            "RMSE": [rmse_tr, rmse_te],
            "MAE": [mae_tr, mae_te],
            "Pearson_r": [pear_tr, pear_te],
            "Spearman_rho": [spear_tr, spear_te],
            "Baseline_RMSE": [np.nan, rmse_baseline],
            "Normalized_RMSE_%": [np.nan, nrmse_pct],
        }
    )
    metrics_df.to_excel(out_dir / "metrics.xlsx", index=False)

    plot_loss(hist, out_dir / "loss_curve.png")
    plot_true_vs_pred(y_test, yhat_test, out_dir / "test_pred_vs_true.png")
    plot_error_hist(y_test, yhat_test, out_dir / "test_error_hist.png")


if __name__ == "__main__":
    main()
