import numpy as np
from data_engineering.parsing import parse_data
from data_engineering.image_loader import build_path_column, create_data_generators
from model.architecture import build_model
from model.training import train
from model.evaluation import evaluate


# Configurações Básicas
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
OUTPUT_DIR = "./output"
SAMPLE_FRACTION = 0.05  # Apenas 5% dos dados para ser mais rápido (1.0 para usar tudo)


def main():
    # 1 - Parse do CSV e split train/val/test (por ID do paciente)
    train_df, val_df, test_df = parse_data()
    
    num_classes = len(train_df['target_vector'].iloc[0])
    
    unique_labels = set()
    for label in train_df['Finding Labels'].unique():
        unique_labels.update(label.split("|"))
    class_names = sorted(unique_labels, key=lambda x: (x != "No Finding", x))
    
    print(f"Classes ({num_classes}): {class_names}")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    
    # Subsample dataset intelligently if testing locally
    if SAMPLE_FRACTION < 1.0:
        print(f"\n[INFO] Smart physical balancing (keeping rare diseases) to {SAMPLE_FRACTION*100}% for fast CPU testing...")
        
        def smart_sample(df, subset_size):
            if len(df) <= subset_size: return df
            labels = np.array(df['target_vector'].tolist())
            class_counts = labels.sum(axis=0)
            rarity = 1.0 / (class_counts + 1)
            # Give each row a sampling priority weight based on its rarest disease
            row_weights = np.max(labels * rarity, axis=1) 
            return df.sample(n=subset_size, weights=row_weights, random_state=42)
            
        train_df = smart_sample(train_df, int(len(train_df) * SAMPLE_FRACTION))
        val_df = smart_sample(val_df, int(len(val_df) * SAMPLE_FRACTION))
        test_df = smart_sample(test_df, int(len(test_df) * SAMPLE_FRACTION))
        print(f"Subset Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    
    # 2 - Add image file paths to each dataframe
    print("\n" + "=" * 60)
    print("STEP 2: Resolving image paths")
    print("=" * 60)
    train_df = build_path_column(train_df)
    val_df = build_path_column(val_df)
    test_df = build_path_column(test_df)
    
    # 3 - Create data generators
    print("\n" + "=" * 60)
    print("STEP 3: Creating data generators")
    print("=" * 60)
    train_gen, val_gen, test_gen = create_data_generators(
        train_df, val_df, test_df, class_names,
        image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
    )
    
    # 4 - Build the model
    print("\n" + "=" * 60)
    print("STEP 4: Building DenseNet121 model")
    print("=" * 60)
    model = build_model(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        num_classes=num_classes,
    )
    
    # 5 - Calculate class weights to deal with class imbalance
    labels_array = np.array(train_df['target_vector'].tolist())
    total_samples = labels_array.shape[0]
    class_counts = labels_array.sum(axis=0)
    
    class_weights = {}
    for i, count in enumerate(class_counts):
        class_weights[i] = total_samples / (num_classes * (count if count > 0 else 1))
        
    print("\n" + "=" * 60)
    print("STEP 5: Training")
    print("=" * 60)
    print("Computed Class Weights (boosting rare diseases):")
    for i, name in enumerate(class_names):
        print(f"  {name:<25}: {class_weights[i]:.2f}")
        
    history = train(
        model, train_gen, val_gen, 
        epochs=EPOCHS, 
        output_dir=OUTPUT_DIR,
        class_weights=class_weights
    )
    
    # 6 - Evaluate on the test set
    print("\n" + "=" * 60)
    print("STEP 6: Evaluating on test set")
    print("=" * 60)
    auc_scores = evaluate(model, test_gen, class_names, output_dir=OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("DONE! Model saved to ./output/best_model.keras")
    print("Use: python predict.py --image <path_to_xray.png>")
    print("=" * 60)


if __name__ == "__main__":
    main()
