# Import packages
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import xgboost as xgb
import numpy as np
from math import ceil
import pickle
from tqdm import tqdm

# Settings
seed = 42
cv_folds = 5
cv_folds_gridsearch = 3
conditions = ["(1) Hypertensive diseases", "(2) Ischaemic heart diseases", "(3) Chronic ischaemic heart disease", "(4) Cardiomyopathies diseases", "(5) Dysrhythmias diseases", "(6) Heart failure"]
# conditions = ["(6) Heart failure"]

# Folder paths
model_folder = "/data/wolf6245/src/HAIM/data/haim_mimiciv/models"
folder_path_embeddings = "/data/wolf6245/src/HAIM/data/haim_mimiciv/embedding/"
y_file = '/data/wolf6245/src/mm_study/data/f_modelling/03_model_input/data-2024-12-19-01-23-23/(3) Chronic ischaemic heart disease/y_fusion_label_not_gt.parquet'


# Load files
def load_data(fname):
    # Get hadm_id
    pickle_file = pd.read_pickle(fname.replace('embedding', 'pickle').replace('.csv', '.pkl'))
    hadm_id = pickle_file.admissions.hadm_id.values
    assert len(hadm_id) == 1, f"hadm_id is not unique: {hadm_id}"
    hadm_id = int(hadm_id[0])

    # Read df
    df = pd.read_csv(fname)
    df["hadm_id"] = hadm_id
    df = df.drop(['img_id', 'img_charttime', 'img_deltacharttime', 'discharge_location', 'img_length_of_stay', 'death_status'], axis = 1)
    return df


def train_model(x_y, folds, model_folder_condition):
    print("--------------------------------------")
    auc_list_folds = []
    best_params = None
    for fold_id, fold in folds.items():
        print(f"---Processing fold {fold_id}...")

        # Get data
        print("----Getting data...")
        train_ids = fold['train_idx']
        test_ids = fold['test_idx']
        df_train = x_y[x_y['hadm_id'].isin(train_ids)]
        df_test = x_y[x_y['hadm_id'].isin(test_ids)]
        y_train = df_train['y']
        y_test = df_test['y']
        x_train = df_train.drop(['y', 'haim_id', 'hadm_id'], axis=1)
        x_test = df_test.drop(['y', 'haim_id', 'hadm_id'], axis=1)
        print(f"hadm_ids in train set: {len(train_ids)}")
        print(f"hadm_ids in test set: {len(test_ids)}")
        print('train, test shapes', x_train.shape, x_test.shape, y_train.shape, y_test.shape)
        print('train set, outcome case = %s, percentage = %s' % (y_train.sum(), y_train.sum()/len(y_train)))
        print('test set, outcome case = %s, percentage = %s' % (y_test.sum(), y_test.sum()/len(y_test)))

        # Train model
        print("----Training model...")
        if best_params is not None:
            print("----Using best settings...")
            param_grid = {k: [v] for k, v in best_params.items()}
        else:
            print("----Using grid search...")
            param_grid = {
                # 'max_depth': [5],
                'max_depth': [5, 7, 8],
                'n_estimators': [200, 300],
                'learning_rate': [0.3],#, 0.1, 0.05],
            }
        gs_metric = 'roc_auc'
        est = xgb.XGBClassifier(verbosity=0, scale_pos_weight = (len(y_train) - sum(y_train))/sum(y_train), seed = seed, eval_metric='logloss', use_label_encoder=False)
        gs = GridSearchCV(estimator = est, param_grid=param_grid, scoring=gs_metric, cv=cv_folds_gridsearch)
        gs.fit(x_train, y_train)

        # Get feature importance
        print("----Getting feature importance...")
        feature_importance = gs.best_estimator_.feature_importances_
        feature_importance = pd.DataFrame(feature_importance, index=x_train.columns, columns=['importance'])
        feature_importance = feature_importance.sort_values(by='importance', ascending=False)
        feature_importance.to_csv(f"{model_folder_condition}/feature_importance_{fold_id}.csv")
        print(f"Feature importance saved.")

        # Store best settings
        if best_params is None:
            print("----Storing best settings...")
            best_params = gs.best_params_
            print(f"Best parameters: {best_params}")

        # Get predictions
        print("----Getting predictions...")
        y_pred_prob_train = gs.predict_proba(x_train)
        y_pred_train = gs.predict(x_train)
        y_pred_prob_test = gs.predict_proba(x_test)
        y_pred_test = gs.predict(x_test)
        y_pred_prob_train = y_pred_prob_train[:, 1]
        y_pred_prob_test = y_pred_prob_test[:, 1]

        # Get metrics
        # f1_train = metrics.f1_score(y_train, y_pred_train, average='macro')
        # accu_train = metrics.accuracy_score(y_train, y_pred_train)
        # accu_bl_train = metrics.balanced_accuracy_score(y_train, y_pred_train)
        if y_train.sum() == 0:
            auc_train = np.nan
        else:
            auc_train =  metrics.roc_auc_score(y_train, y_pred_prob_train)
        # print(f'F1 Score for Training Set is: {f1_train}')
        # print(f'Accuracy for Training Set is: {accu_train}')
        # print(f'Balanced Accuracy for Training Set is: {accu_bl_train}')
        print(f'AUC for Training Set is: {auc_train}')
        # f1_test = metrics.f1_score(y_test, y_pred_test, average='macro')
        # accu_test = metrics.accuracy_score(y_test, y_pred_test)
        # accu_bl_test = metrics.balanced_accuracy_score(y_test, y_pred_test)
        if y_test.sum() == 0:
            auc_test = np.nan
        else:
            auc_test =  metrics.roc_auc_score(y_test, y_pred_prob_test)
        # print(f'F1 Score for Testing Set is: {f1_test}')
        # print(f'Accuracy for Testing Set is: {accu_test}')
        # print(f'Balanced Accuracy for Testing Set is: {accu_bl_test}')
        print(f'AUC for Testing Set is: {auc_test}')
        auc_list_folds.append(auc_test)
        print("--------------------------------------")

    return auc_list_folds


if __name__ == "__main__":

    np.random.seed(seed)
    os.makedirs(model_folder, exist_ok=True)

    # Load y data
    print("------Loading y data...------")
    df_y = pd.read_parquet(y_file)
    df_y.hadm_id = df_y.hadm_id.astype(int)
    print(f"Loaded y data with shape: {df_y.shape}")

    # Load all the pickle files
    print("------Loading all pickle files...------")
    all_pickle_files = [f"{folder_path_embeddings}{f}" for f in os.listdir(folder_path_embeddings) if f.endswith('.csv')]
    all_pickle_files.sort()
    print(f"Found {len(all_pickle_files)} pickle files in {folder_path_embeddings}")
    df_list = []
    first_columns_list = []
    for fname in tqdm(all_pickle_files):
        df_aux = load_data(fname)
        columns_list_new = df_aux.columns.tolist()
        if len(first_columns_list) == 0:
            first_columns_list = columns_list_new
        else:
            if set(first_columns_list) != set(columns_list_new):
                print(f"Columns in {fname} do not match previous files")
        df_list.append(df_aux)
    all_columns = [set(df.columns) for df in df_list]
    common_columns = sorted(set.intersection(*all_columns))
    not_in_all_columns = sorted(set.union(*all_columns) - set(common_columns))
    print("Columns not in all DataFrames:", not_in_all_columns)
    df_list_filtered = [df[common_columns] for df in df_list]
    df = pd.concat(df_list_filtered, ignore_index=True)
    print(f"Loaded all data with shape: {df.shape}")
    columns_to_drop = [
        'split', 'Atelectasis', 'Cardiomegaly', 'Consolidation',
        'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
        'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
        'Pneumonia', 'Pneumothorax', 'Support Devices',
        'PerformedProcedureStepDescription', 'ViewPosition',
    ]
    df_X = df.drop(columns=columns_to_drop).copy()
    df_X = df_X[~df_X.isna().any(axis=1)]
    print(f"Using df_X data after filtering na: {df_X.shape}")

    # Merge labels
    df_y_aux = df_y[['hadm_id'] + conditions].copy()
    x_y = df_X.merge(df_y_aux, on='hadm_id', how='left')
    # x_y["y"] = np.random.choice([0, 1], size=len(x_y), p=[0.5, 0.5])

    # Get splits
    print("------Getting splits...------")
    pkl_list = x_y['hadm_id'].unique().tolist()
    np.random.shuffle(pkl_list)
    total_num = len(pkl_list)
    print('Total number of oberservations = %s' % total_num)
    folds = {}
    fold_size = ceil(total_num / cv_folds)
    for i in range(cv_folds):
        start_id_test = i * fold_size
        end_id_test = min(start_id_test + fold_size, total_num)
        test_ids = pkl_list[start_id_test:end_id_test]
        train_ids = pkl_list[:start_id_test] + pkl_list[end_id_test:]
        folds[str(i)] = {
            'train_idx': train_ids,
            'test_idx': test_ids,
        }
    
    # Loop through conditions
    auc_dict = {}
    for condition in conditions:
        print(f"------Processing condition: {condition}...------")
        model_folder_condition = os.path.join(model_folder, condition.replace(" ", "_"))
        os.makedirs(model_folder_condition, exist_ok=True)
        
        # Get conditions as y
        print("--Getting conditions...")
        x_y_condition = x_y.copy()
        x_y_condition["y"] = x_y_condition[condition].astype(int)
        x_y_condition = x_y_condition.drop(columns=conditions)
        assert x_y_condition.shape[0] == x_y.shape[0], f"Shape mismatch after merge: {x_y_condition.shape[0]} vs {x_y_shape[0]}"
        assert x_y_condition.y.notna().sum() == x_y_condition.shape[0], f"y is not complete: {x_y_condition.y.notna().sum()} vs {x_y_condition.shape[0]}"

        # Run models
        print("--Running models...")
        auc_list_condition = train_model(
            x_y=x_y_condition,
            folds=folds,
            model_folder_condition=model_folder_condition,
        )
        auc_dict[condition] = auc_list_condition
        print(f"Finished processing condition: {condition}")
        print(f"AUC list: {auc_list_condition}")
        mean_auc = np.mean(auc_list_condition)
        std_auc = np.std(auc_list_condition)
        print(f"Mean AUC: {mean_auc}")
        print(f"Standard deviation of AUC: {std_auc}")
        print("---------------------------")
        
        # Save results as text file
        with open(f"{model_folder}/auc_list_{condition}.txt", "w") as f:
            f.write(f"AUC list: {auc_list_condition}\n")
            f.write(f"Mean AUC: {mean_auc}\n")
            f.write(f"Standard deviation of AUC: {std_auc}\n")
        print("--------------------------------------------------------------")
    
    # Save all results
    with open(f"{model_folder}/auc_dict.txt", "w") as f:
        all_conditions = []
        for condition, auc_list in auc_dict.items():
            f.write(f"Condition: {condition}\n")
            f.write(f"AUC list: {auc_list}\n")
            mean_auc = np.mean(auc_list)
            std_auc = np.std(auc_list)
            f.write(f"Mean AUC: {mean_auc}\n")
            f.write(f"Standard deviation of AUC: {std_auc}\n")
            all_conditions.extend(auc_list)
        overall_mean_auc = np.mean(all_conditions)
        overall_std_auc = np.std(all_conditions)
        f.write(f"Overall mean AUC: {overall_mean_auc}\n")
        f.write(f"Overall standard deviation of AUC: {overall_std_auc}\n")
        print("--------------------------------------------------------------")
        print(f"Overall mean AUC: {overall_mean_auc}")
        print(f"Overall standard deviation of AUC: {overall_std_auc}")
    print("Done. Finished processing all conditions.")
