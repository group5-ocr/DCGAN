import os, math, numpy as np, tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras import Model, Sequential
# ------------------------------------------------------------
# 설정
# ------------------------------------------------------------
IMG_SHAPE = (28, 28, 1)     # 데이터 모양 (MNIST 흑백 이미지)
Z_DIM     = 100             # 노이즈 차원 (입력)
BATCH     = 128
EPOCHS    = 50
LR        = 2e-4
BETA1     = 0.5
SEED      = 42

# 랜덤 시드 고정 (재현성 ↑)
tf.random.set_seed(SEED)
np.random.seed(SEED)

# MNIST 데이터셋 로딩 & 전처리
def load_dataset():
    (x_train, _), _ = tf.keras.datasets.mnist.load_data()
    x = x_train.astype("float32")
    x = np.expand_dims(x, axis=-1)         # (N, 28, 28, 1)
    x = (x / 127.5) - 1.0                  # [0,255] -> [-1,1]
    ds = tf.data.Dataset.from_tensor_slices(x)
    ds = ds.shuffle(60000, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return ds

# Generator (생성자)
def build_generator(z_dim=Z_DIM):
    # 28x28x1을 만드는 간단한 DCGAN 제너레이터
    g = Sequential(name="generator")
    g.add(L.Input(shape=(z_dim,)))
    g.add(L.Dense(7*7*128, use_bias=False))
    g.add(L.BatchNormalization())
    g.add(L.LeakyReLU(0.2))
    g.add(L.Reshape((7, 7, 128)))
    g.add(L.Conv2DTranspose(64, kernel_size=5, strides=2, padding="same", use_bias=False))
    g.add(L.BatchNormalization())
    g.add(L.LeakyReLU(0.2))
    g.add(L.Conv2DTranspose(1, kernel_size=5, strides=2, padding="same", activation="tanh"))
    return g

# Discriminator (판별자)
def build_discriminator(img_shape=IMG_SHAPE):
    d = Sequential(name="discriminator")
    d.add(L.Input(shape=img_shape))
    d.add(L.Conv2D(64, 5, strides=2, padding="same"))
    d.add(L.LeakyReLU(0.2))
    d.add(L.Dropout(0.3))
    d.add(L.Conv2D(128, 5, strides=2, padding="same"))
    d.add(L.LeakyReLU(0.2))
    d.add(L.Dropout(0.3))
    d.add(L.Flatten())
    d.add(L.Dense(1, activation="sigmoid"))
    return d

g = build_generator()
d = build_discriminator()

# Discriminator 학습 설정
d.trainable = True
opt_d = tf.keras.optimizers.Adam(LR, beta_1=BETA1)
d.compile(optimizer=opt_d, loss="binary_crossentropy", metrics=["accuracy"])

# Combined 모델 (G + frozen D)
d.trainable = False
z_in = L.Input(shape=(Z_DIM,))
valid = d(g(z_in))
combined = Model(z_in, valid, name="G_with_frozen_D")
opt_g = tf.keras.optimizers.Adam(LR, beta_1=BETA1)
combined.compile(optimizer=opt_g, loss="binary_crossentropy")

# (참고) 여기서 d.trainable을 True로 돌려놔도, d는 이미 train용으로 컴파일된 상태라
# 이후 루프에서 다시 토글/재컴파일할 필요가 전혀 없습니다.
# 다시 D는 단독 학습을 위해 trainable True로 돌려놓음
d.trainable = True
# ------------------------------------------------------------
# 학습 루프 (루프 안에서는 절대 trainable 토글 금지!)
# ------------------------------------------------------------
def smooth_labels(y, low=0.9, high=1.0):
    # one-sided label smoothing (real만)
    return tf.random.uniform(tf.shape(y), minval=low, maxval=high)

def maybe_flip(y, p=0.03):
    # 작은 확률로 라벨은 뒤집음
    mask = tf.cast(tf.random.uniform(tf.shape(y)) < p, y.dtype)
    return y * (1 - mask) + (1 - y) * mask

def train(epochs=EPOCHS):
    ds = load_dataset()
    num_batches = None
    for epoch in range(1, epochs + 1):
        d_losses, g_losses = [], []
        for step, real in enumerate(ds, start=1):
            bs = tf.shape(real)[0]
            # 가짜 이미지 생성
            z = tf.random.normal((bs, Z_DIM))
            fake = g(z, training=False)
            # 라벨 생성
            y_real = smooth_labels(tf.ones((bs, 1), dtype=tf.float32), 0.85, 1.0)
            y_fake = tf.zeros((bs, 1), dtype=tf.float32)
            # (optional) flip a small portion
            y_real = maybe_flip(y_real, p=0.02)
            y_fake = maybe_flip(y_fake, p=0.02)
            # Discriminator 학습
            d_loss_real = d.train_on_batch(real, y_real, return_dict=True)
            d_loss_fake = d.train_on_batch(fake, y_fake, return_dict=True)
            d_loss = {
                "loss": 0.5 * (d_loss_real["loss"] + d_loss_fake["loss"]),
                "acc":  0.5 * (d_loss_real["accuracy"] + d_loss_fake["accuracy"]),
            }
            d_losses.append(d_loss["loss"])
            # Generator 학습 (D는 frozen)
            z = tf.random.normal((bs, Z_DIM))
            g_loss = combined.train_on_batch(z, tf.ones((bs, 1)), return_dict=True)
            g_losses.append(g_loss["loss"])
            num_batches = step
        print(f"epoch:{epoch:>3d}  d_loss:{np.mean(d_losses):.4f}  g_loss:{np.mean(g_losses):.4f}  (steps:{num_batches})")
# 실행
train(EPOCHS)

# 학습된 Genartor로 샘플 이미지 생성
def sample_images(rows=5, cols=5):
    import matplotlib.pyplot as plt
    z = tf.random.normal((rows*cols, Z_DIM))
    gen = g.predict(z, verbose=0)
    # [-1,1] -> [0,1]
    gen = (gen + 1.0) / 2.0

    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
    for i, ax in enumerate(axes.flat):
        img = gen[i, :, :, 0]
        ax.imshow(img, cmap="gray")
        ax.axis("off")
    plt.tight_layout()
    os.makedirs("samples", exist_ok=True)
    plt.savefig("samples/grid.png", dpi=150)
    plt.close()
# 실행
sample_images()