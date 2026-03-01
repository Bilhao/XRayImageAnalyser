import pandas as pd
from sklearn.model_selection import train_test_split


def labels_to_list_int(df, column_name):
    unique_labels = set()
    for label in df[column_name].unique():
        unique_labels.update(label.split("|"))
        
    sorted_labels = sorted(unique_labels, key=lambda x: (x != "No Finding", x))
    label_to_int = {label: idx for idx, label in enumerate(sorted_labels)}
    
    def map_labels(x):
        row_list = [0] * len(label_to_int)
        for label in x.split("|"):
            if label in label_to_int:
                row_list[label_to_int[label]] = 1
        return row_list
    
    df['target_vector'] = df[column_name].apply(map_labels)
    return df


def parse_data():
    df = pd.read_csv("./data/Data_Entry_2017.csv")    
    df = labels_to_list_int(df, 'Finding Labels')
    
    df['Patient Gender'] = df['Patient Gender'].map({'M': 0, 'F': 1})
    df['View Position'] = df['View Position'].map({'PA': 0,'AP': 1})
    
    df = df[df['Patient Age'] <= 100]
    df = df.drop(columns=['Unnamed: 11'])

    patient_ids = df['Patient ID'].unique()
    train_ids, temp_ids = train_test_split(patient_ids, test_size=0.20, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=42)
    
    train_df = df[df["Patient ID"].isin(train_ids)]
    val_df = df[df["Patient ID"].isin(val_ids)]
    test_df = df[df["Patient ID"].isin(test_ids)]
    
    return train_df, val_df, test_df

