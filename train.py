import tensorflow as tf


def train_model(
    model,
    X_train,
    X_val,
    epochs=8,
    batch_size=512,
    checkpoint_path=None
):
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=1,
            min_lr=1e-6,
            verbose=1
        )
    ]

    if checkpoint_path:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor="val_loss",
                save_best_only=True,
                verbose=1
            )
        )

    return model.fit(
        X_train,
        X_train,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=1
    )
