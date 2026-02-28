import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import preprocess_input


def find_image_path(image_name, base_dir="./data"):
    for i in range(1, 13):
        folder = os.path.join(base_dir, f"images_{i:03d}", "images")
        full_path = os.path.join(folder, image_name)
        if os.path.exists(full_path):
            return full_path
    return None


def build_path_column(df, base_dir="./data"):
    df['full_path'] = df['Image Index'].apply(lambda x: find_image_path(x, base_dir))
    
    missing = df['full_path'].isna().sum()
    if missing > 0:
        print(f"{missing} images not found on disk. Dropping these rows.")
        df = df.dropna(subset=['full_path'])
    
    print(f"Images with valid paths: {len(df)}")
    return df


def create_data_generators(train_df, val_df, test_df, class_names, image_size=224, batch_size=32):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        horizontal_flip=True,
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
    )
    
    val_test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
    )
    
    def _make_generator(df, datagen, shuffle):
        labels = np.array(df['target_vector'].tolist())
        label_df = pd.DataFrame(labels, columns=class_names, index=df.index)
        
        gen_df = df[['full_path']].copy()
        gen_df = gen_df.join(label_df)
        
        return datagen.flow_from_dataframe(
            dataframe=gen_df,
            x_col='full_path',
            y_col=class_names,
            target_size=(image_size, image_size),
            batch_size=batch_size,
            class_mode='raw',
            color_mode='rgb',
            shuffle=shuffle,
        )
    
    train_gen = _make_generator(train_df, train_datagen, shuffle=True)
    val_gen = _make_generator(val_df, val_test_datagen, shuffle=False)
    test_gen = _make_generator(test_df, val_test_datagen, shuffle=False)
    
    print(f"Train batches: {len(train_gen)} | Val batches: {len(val_gen)} | Test batches: {len(test_gen)}")
    return train_gen, val_gen, test_gen


if __name__ == "__main__":
    from parsing import main as parse_main
    
    train_df, val_df, test_df = parse_main()
    
    train_df = build_path_column(train_df)
    val_df = build_path_column(val_df)
    test_df = build_path_column(test_df)
    
    num_classes = len(train_df['target_vector'].iloc[0])
    print(f"Number of classes: {num_classes}")
