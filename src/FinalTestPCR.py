import pandas as pd
import numpy as np
import joblib
import pickle
import warnings

warnings.filterwarnings('ignore')


def load_preprocessing_objects():
    model = joblib.load('xgb_pcr_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_selector = joblib.load('feature_selector.pkl')

    with open('selected_features.pkl', 'rb') as f:
        selected_features = pickle.load(f)

    with open('encoding_mappings.pkl', 'rb') as f:
        encoding_mappings = pickle.load(f)

    with open('important_features.pkl', 'rb') as f:
        important_features = pickle.load(f)

    with open('training_feature_names.pkl', 'rb') as f:
        training_feature_names = pickle.load(f)

    return model, scaler, feature_selector, selected_features, encoding_mappings, important_features, training_feature_names


def preprocess_test_data(test_data, encoding_mappings):
    test_data_processed = test_data.copy()

    for col, mapping_info in encoding_mappings.items():
        if col in test_data_processed.columns:
            known_classes = mapping_info['classes']
            class_to_int = {cls: idx for idx, cls in enumerate(known_classes)}

            test_data_processed[col] = (
                test_data_processed[col].astype(str)
                .apply(lambda x: class_to_int.get(x, 0))
            )

    return test_data_processed


def predict_pcr(test_file_path):

    print("Starting PCR prediction...")

    model, scaler, feature_selector, selected_features, encoding_mappings, important_features, training_feature_names = load_preprocessing_objects()

    test_data = pd.read_excel(test_file_path)    

    patient_ids = (
        test_data['ID'].values
        if 'ID' in test_data.columns else
        np.arange(1, len(test_data) + 1)
    )

    drop_cols = ['ID', 'pCR (outcome)', 'RelapseFreeSurvival (outcome)']
    X_test = test_data.drop(columns=[c for c in drop_cols if c in test_data.columns])

    X_test_encoded = preprocess_test_data(X_test, encoding_mappings)

    X_test_aligned = pd.DataFrame(index=X_test_encoded.index)

    for feat in training_feature_names:
        X_test_aligned[feat] = X_test_encoded.get(feat, 0)

    X_test_scaled = scaler.transform(X_test_aligned)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=training_feature_names)

    X_test_final = X_test_scaled_df[selected_features]

    predictions = model.predict(X_test_final)

    predictions_df = pd.DataFrame({
        'ID': patient_ids,
        'pCR (outcome)': predictions
    })

    predictions_df.to_csv('PCRPrediction.csv', index=False)

    print("Done. Results saved to PCRPrediction.csv")

    return predictions_df


if __name__ == "__main__":
    import sys
    import glob

    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        test_files = glob.glob('FinalTestDataset2025.xls*')
        if not test_files:
            print("No test dataset found.")
            sys.exit(1)
        test_file = test_files[0]

    predict_pcr(test_file)
