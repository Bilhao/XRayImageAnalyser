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
SAMPLE_FRACTION = 1.0  # Apenas 5% dos dados para ser mais rápido (1.0 para usar tudo)


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
    
    # Corta o dataset para treinar mais rápido, mas tenta manter a distribuição de doenças para não perder as raras
    if SAMPLE_FRACTION < 1.0:        
        def cut_sample(df, subset_size):
            if len(df) <= subset_size: return df
            labels = np.array(df['target_vector'].tolist())
            class_counts = labels.sum(axis=0)
            rarity = 1.0 / (class_counts + 1)
            # Dá a cada linha um peso de prioridade baseado na raridade das doenças presentes
            row_weights = np.max(labels * rarity, axis=1) 
            return df.sample(n=subset_size, weights=row_weights, random_state=42)
            
        train_df = cut_sample(train_df, int(len(train_df) * SAMPLE_FRACTION))
        val_df = cut_sample(val_df, int(len(val_df) * SAMPLE_FRACTION))
        test_df = cut_sample(test_df, int(len(test_df) * SAMPLE_FRACTION))
        print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    
    # 2 - Adiciona o caminho completo das imagens nos dataframes
    train_df = build_path_column(train_df)
    val_df = build_path_column(val_df)
    test_df = build_path_column(test_df)
    
    # 3 - Create data generators
    train_gen, val_gen, test_gen = create_data_generators(
        train_df, val_df, test_df, class_names,
        image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
    )
    
    # 4 - Constrói o modelo
    model = build_model(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        num_classes=num_classes,
    )
    
    # 5 - Calcula o peso de cada classe para lidar com o desbalanceamento das doenças raras
    labels_array = np.array(train_df['target_vector'].tolist())
    total_samples = labels_array.shape[0]
    class_counts = labels_array.sum(axis=0)
    
    class_weights = {}
    for i, count in enumerate(class_counts):
        class_weights[i] = total_samples / (num_classes * (count if count > 0 else 1))
        
    print("Computed Class Weights (boosting rare diseases):")
    for i, name in enumerate(class_names):
        print(f"  {name:<25}: {class_weights[i]:.2f}")
        
    train(
        model, train_gen, val_gen, 
        epochs=EPOCHS, 
        output_dir=OUTPUT_DIR,
        class_weights=class_weights
    )
    
    evaluate(model, test_gen, class_names, output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()
