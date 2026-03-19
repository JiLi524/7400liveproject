from pathlib import Path
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def main():
    data_path = Path('results') / 'features_dataset_pass_only_clean_complete.csv'
    if not data_path.exists():
        raise FileNotFoundError(f'Dataset not found: {data_path.resolve()}')

    model_df = pd.read_csv(data_path)
    feature_cols = [c for c in model_df.columns if c.startswith('f_')]
    X = model_df[feature_cols].copy()
    y = model_df['is_fruit'].astype(int).copy()

    svm_model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', probability=True, random_state=42))
    ])

    svm_model.fit(X, y)
    joblib.dump(svm_model, 'svm_pipeline.joblib')
    print('Saved model to svm_pipeline.joblib')
    print(f'Number of features used: {len(feature_cols)}')


if __name__ == '__main__':
    main()
