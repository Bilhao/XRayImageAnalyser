import argparse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.densenet import preprocess_input


DEFAULT_CLASS_NAMES = [
    "No Finding", "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia", "Infiltration",
    "Mass", "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]
NUM_PREDICTIONS = 5


def predict_single(model_path, image_path, class_names=None, image_size=224):
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES
    
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    
    img = load_img(image_path, target_size=(image_size, image_size))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    predictions = model.predict(img_array, verbose=0)[0]
    
    sorted_indices = np.argsort(predictions)[::-1]
    
    print(f"\nPredictions for: {image_path}")
    print(f"{'Pathology':<25} {'Confidence':>10}")
    print("-" * 37)
    
    for i in range(NUM_PREDICTIONS):
        idx = sorted_indices[i]
        print(f"{class_names[idx]:<25} {predictions[idx]*100:>10.4f}%")
    
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model", type=str, default="./output/best_model.keras")
    args = parser.parse_args()
    
    predict_single(args.model, args.image)
