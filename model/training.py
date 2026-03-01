import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


def train(model, train_gen, val_gen, epochs=20, output_dir="./output", class_weights=None):
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(output_dir, "best_model.keras"),
        monitor='val_loss',
        save_best_only=True,
        verbose=1,
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    )
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[checkpoint, early_stop, reduce_lr],
        class_weight=class_weights,
    )
    
    plot_history(history, output_dir)
    return history


def plot_history(history, output_dir="./output"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Perdas
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Binary Crossentropy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Acurácia
    axes[1].plot(history.history['binary_accuracy'], label='Train Accuracy')
    axes[1].plot(history.history['val_binary_accuracy'], label='Val Accuracy')
    axes[1].set_title('Binary Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "training_history.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
