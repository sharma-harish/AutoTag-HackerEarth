import category_encoders as ce
import joblib
import os
import pandas as pd
from sklearn import model_selection

TRAIN_CSV=os.environ.get('TRAIN_CSV')
KFOLD_CSV=os.environ.get('KFOLD_CSV')

if __name__ == "__main__":
    df = pd.read_csv(TRAIN_CSV)
    df['kfold'] = -1
    
    df = df.sample(frac = 1).reset_index(drop=True)
    print(df.Class.value_counts())
    kf = model_selection.StratifiedKFold(n_splits=10, shuffle=False, random_state=42)

    mapping = [{'col': 'Class', 'mapping': {'Food': 0, 'Attire': 1, 'misc': 2, 'Decorationandsignage': 3}}]
    encoder = ce.ordinal.OrdinalEncoder(cols=['Class'], mapping = mapping)
    df = encoder.fit_transform(df)
    joblib.dump(encoder, f'F:\Workspace\AutoTag-HE\models\y_encoder.pkl')
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X = df, y = df.Class.values)):
        print(len(train_idx), len(val_idx))
        df.loc[val_idx, 'kfold'] = fold
        
    print(df.head())
    df.to_csv(KFOLD_CSV, index=False)