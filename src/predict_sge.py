'''
   Predict Spatial Gene Expression from H&E

'''

import argparse
import datetime
import glob
import os
import numpy as np
import pandas as pd
import time

from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns


from visualize import visualise_feature_vector_heatmap, viz_side_by_side, visualise_feature_matrix_heatmap


from utils import set_seed, save_features_with_names, load_features_from_csv
from feature_extractor import huggingfaceLogin, getFeatureMap, transformImage, extractFeatures
from feature_extractor import extractFeatureVectorUsingUNI, extractFeatureVectorUsingGigaPath


from sklearn.model_selection import train_test_split, GroupShuffleSplit, KFold, GroupKFold, GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import f1_score, roc_curve, roc_auc_score,recall_score,precision_score,accuracy_score, log_loss

from scipy.stats import loguniform, uniform, randint, pearsonr, spearmanr


from huggingface_hub import login


SCORE_STAT = 'neg_mean_squared_error'

def plot_roc_curve(fpr,tpr,auc,title='ROC Curve'):
    plt.figure(figsize=(5, 5))
    # Plot tpr against fpr
    plt.plot(fpr, tpr,label="auc="+str(auc))
    #control line
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.show()

def get_test_scores(y_test, y_pred,y_pred_probs,train_score=0.0,label=""):
    f1 = f1_score(y_test, y_pred , average="macro")
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_pred_probs)

    fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
    auc = roc_auc_score(y_test,y_pred_probs)
    plot_roc_curve(fpr,tpr,auc, title=f"ROC Curve for {label}")

    data = [(recall, precision, f1, accuracy, auc,train_score,loss)]
    df = pd.DataFrame(data, columns=['Recall', 'Precision', 'F1 Score', 'Accuracy','AUC', 'Train F1','Loss'])
    df.insert(0, 'Run', label)

    return df

def evaluate_model(model, X_test, y_test, label=''):
    y_pred = model.predict(X_test)
    y_pred_probs = model.predict_proba(X_test)[:,1]
    scores = get_test_scores(y_test, y_pred, y_pred_probs,train_score=model.best_score_,label=label )


    return scores


def get_regression_scores_single(y_test, y_pred, train_score=0.0, label=""):
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    pearson_corr, _ = pearsonr(y_test, y_pred)
    spearman_corr, _ = spearmanr(y_test, y_pred)

    # Plot actual vs predicted
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel("True Expression")
    plt.ylabel("Predicted Expression")
    plt.title(f"Actual vs Predicted for {label}")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.grid(True)
    plt.show()

    data = [(mse, rmse, mae, r2, pearson_corr, spearman_corr, train_score)]
    df = pd.DataFrame(data, columns=[
        'MSE', 'RMSE', 'MAE', 'R2',
        'Pearson Corr', 'Spearman Corr',
        'Train CV Score'
    ])
    df.insert(0, 'Run', label)
    return df

def evaluate_regression_model_single(model, X_test, y_test, label=''):
    y_pred = model.predict(X_test)
    y_pred = y_pred.flatten() if hasattr(y_pred, 'shape') and len(y_pred.shape) > 1 else y_pred
    scores = get_regression_scores_single(y_test, y_pred, train_score=model.best_score_, label=label)
    return scores

"""updated for mean of multioutput"""

def get_regression_scores(y_test, y_pred, train_score=0.0, label=""):
    gene_names = list(y_test.columns)
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)

    for i, gene in enumerate(gene_names):
        print(f"SCORING {gene}")
        print(f"y_test: {y_test[:,i]}")
        print(f"y_pred: {y_pred[:,i]}")

    # Compute per-gene (column) metrics
    mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
    r2 = r2_score(y_test, y_pred, multioutput='raw_values')

    # Correlations per gene
    pearson_corrs = [pearsonr(y_test[:, i], y_pred[:, i])[0] for i in range(y_test.shape[1])]
    spearman_corrs = [spearmanr(y_test[:, i], y_pred[:, i])[0] for i in range(y_test.shape[1])]

    # Per-gene DataFrame
    n_genes = y_test.shape[1]
    per_gene_df = pd.DataFrame({
        'Gene': gene_names, #[f'Gene_{i}' for i in range(n_genes)],
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Pearson Corr': pearson_corrs,
        'Spearman Corr': spearman_corrs
    })
    per_gene_df.insert(0, 'Run', label)

    # Summary row
    # Aggregate (mean)
    mean_metrics = {
        'MSE': mse.mean(),
        'RMSE': rmse.mean(),
        'MAE': mae.mean(),
        'R2': r2.mean(),
        'Pearson Corr': np.nanmean(pearson_corrs),
        'Spearman Corr': np.nanmean(spearman_corrs),
        'Train CV Score': train_score
    }

    summary_df = pd.DataFrame([mean_metrics])
    summary_df.insert(0, 'Run', label)


    curr_date=str(datetime.date.today())
    curr_time=datetime.datetime.now().strftime('%H%M')
    filename=f'summaryplot-gene0-{label}-{curr_date}_{curr_time}.png'

    # Optional: Plot one gene as example (e.g., gene 0)
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_test[:, 0], y=y_pred[:, 0])
    plt.xlabel("True Expression (Gene 0)")
    plt.ylabel("Predicted Expression")
    plt.title(f"Actual vs Predicted for {label} (Gene 0)")
    plt.plot([y_test[:, 0].min(), y_test[:, 0].max()],
             [y_test[:, 0].min(), y_test[:, 0].max()], 'r--')
    plt.grid(True)
    plotname = f"scatter_{label}_{gene_names[0]}.png".replace(" ", "_")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    #plt.show()

    return summary_df, per_gene_df



def evaluate_regression_model(models, X_test, y_test, label='', save=True):
    # Predict each gene separately
    y_pred = np.column_stack([model.predict(X_test) for model in models])

    # If you want to average the best CV score across models
    train_scores = [getattr(model, 'best_score_', np.nan) for model in models]
    print(f"train scores: {train_scores}")
    for model in models:
        print(model.best_score_)

    mean_train_score = np.nanmean(train_scores)

    # Compute metrics
    summary_scores, per_gene_scores = get_regression_scores(
        y_test,
        y_pred,
        train_score=mean_train_score,
        label=label
    )

    if save:
        # Save per-gene metrics
        per_gene_scores.to_csv(f"per_gene_scores-{label}.csv", index=False)
        summary_scores.to_csv(f"summary_scores-{label}.csv", index=False)

    return summary_scores, per_gene_scores


def train_multioutput_regressor(base_model, params, label, X_train, y_train, X_test, y_test, gkf, groups):
    tuned_models = []
    print(f"{label.upper()}: Start training")
    for i in range(y_train.shape[1]):
        print(f"Fitting gene {i}")
        grid = GridSearchCV(
            estimator=base_model,
            param_grid=params,
            cv=gkf,
            scoring=SCORE_STAT,
            n_jobs=-1,
            verbose=1
        )
        grid.fit(X_train, y_train.iloc[:, i], groups=groups)
        #tuned_models.append(grid.best_estimator_)
        tuned_models.append(grid)
        print(grid.best_params_)
        print(grid.best_score_)
        print("Best estimator:", grid.best_estimator_)
    print(tuned_models)

    #Evaluate on devel dataset
    summary_scores, per_gene_scores = evaluate_regression_model(tuned_models, X_test, y_test, label)
    print(f"{label.upper()}: Finished")
    return grid, summary_scores


#######################################################################################################

def train(feature_matrix, merged_df, outcome_cols):
    
    """
    Need to keep LEAPIDs together in train/dev split as well as in folds to avoid data leakage
    """

    gss = GroupShuffleSplit(
        n_splits=1,          # just one train/test split
        test_size=0.2,       # 20 % of the groups → test set
        random_state=77)     # reproducible

    train_idx, test_idx = next( gss.split(feature_matrix_clean,              # X
                  y=None,                 # y (optional)
                  groups=merged_df['LEAP_ID']))  # the grouping key

    """Check LEAP_IDs are only in either train or test - should pass the assertion with no output"""
    train_LEAPIDs = set(merged_df['LEAP_ID'].iloc[train_idx].unique())
    test_LEAPIDs = set(merged_df['LEAP_ID'].iloc[test_idx].unique())
    assert train_LEAPIDs.isdisjoint(test_LEAPIDs), (
        f"Data-leakage: these LEAP_IDs are in both splits → "
        f"{train_LEAPIDs & test_LEAPIDs}"
    )

    X_train = feature_matrix_clean[train_idx]
    X_dev  = feature_matrix_clean[test_idx]

    y_train = merged_df[outcome_cols].iloc[train_idx]     # if you have a target vector/series
    y_dev  = merged_df[outcome_cols].iloc[test_idx]

    ################### MULTI-OUTPUT REGRESSION ###################
    seed = 77
    kf_n=5
    ## have to ensure we don't split LEAPIDs between train and validation sets in each fold
    gkf = GroupKFold(n_splits=kf_n) 
    # `groups` must line up 1-for-1 with the rows of X_train / y_train
    groups = merged_df['LEAP_ID'].iloc[train_idx].to_numpy()


    roc_scores = []
    predictions = []
    best_models = {}

    ######### DEFINE MODELS TO TRAIN #################################
    # ( label, model, params )
    model_configs = [
        ("elasticnet", ElasticNet(random_state=seed), {
            'alpha': [0.1, 0.5, 1, 5, 10],
            'l1_ratio': [0, 0.1, 0.5, 0.9, 1],
            'tol': [0.0001, 0.0005, 0.001],
            'max_iter': [100, 200, 500, 1000]
        }),
        ("randomforest", RandomForestRegressor(random_state=seed), {
            'n_estimators': [100],
            'max_depth': [10],
            'min_samples_split': [2],
            'min_samples_leaf': [2]
        }),
        ("gradboost", HistGradientBoostingRegressor(random_state=seed), {
            'learning_rate': [0.01, 0.05],
            'max_iter': [300],
            'max_depth': [5, None],
            'min_samples_leaf': [10, 20, 50],
            'l2_regularization': [0, 0.1, 1.0]
        }),
        ("SVM", SVR(), [
            {
                'kernel': ['rbf'],
                'C': [0.01, 0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.01, 0.1, 1],
                'epsilon': [0.01, 0.05, 0.1, 0.2, 0.5]
            },
            {
                'kernel': ['poly', 'sigmoid'],
                'C': [0.01, 0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.01, 0.1, 1],
                'degree': [2, 3, 4],
                'coef0': [0.0, 0.1, 0.5, 1],
                'epsilon': [0.01, 0.05, 0.1, 0.2, 0.5]
            }
        ])
    ]

    
    ######### FOR TESTING ONLY
    #model_configs = [
    #    ("SVM", SVR(), 
    #        {
    #            'kernel': ['rbf'],
    #            'C': [0.01, 0.1, 1, 10],
    #            'gamma': ['scale', 'auto', 0.01, 0.1, 1],
    #            'epsilon': [0.01, 0.05, 0.1, 0.2, 0.5]
    #        }
    #    )]

    ######### REMOVE AFTER TESTING

    print("starting training loop")
    ######### TRAIN MODELS ###########################################
    for label, model, params in model_configs:
        grid, scores = train_multioutput_regressor(model, params, label, X_train, y_train, X_dev, y_dev, gkf, groups)
        predictions.append(scores)
        print(scores)
        best_models[label] = grid

    predictions_c = pd.concat(predictions, ignore_index=True, sort=False)
    predictions_c['feature_extractor'] = model_name
    return best_models, predictions_c



if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-ip', '--input_path', required=True, help='path to h&e patches or path to feature file')
    ap.add_argument('-tp', '--truth_path', required=True, help='path and filename of label csv')
    ap.add_argument('-op', '--output_path', required=True, help='path to save output')
    ap.add_argument('-tf', '--truth_file',  required=True, help='name of csv')
    ap.add_argument('-ff', '--feature_file',  help='name of csv')
    ap.add_argument('-s', '--segment', default='PanCK+', help='PanCK+ or CD45+')
    ap.add_argument('-fm', '--foundation_model', default='uni', help='one of uni, gigapath, virchow')
    ap.add_argument('-gf', '--get_features', action='store_true')
    #ap.add_argument('-cp', '--config_path', help='full path to config file')


    args = ap.parse_args()

    #get current date and time for model name
    curr_date=str(datetime.date.today())
    curr_time=datetime.datetime.now().strftime('%H%M')


    name=f"run-{args.segment}_{args.foundation_model}_{curr_date}_{curr_time}"
    print(name)
    #set up paths for models, training curves and predictions
    save_path = os.path.join(args.output_path,name)
    os.makedirs(save_path,exist_ok=True)

    # READ CONFIG
    #if args.config_path:
    #    with open(args.config_path, "r") as file:
    #        config = yaml.safe_load(file)
    #else:
    #    config = None



    print(f"ip: {args.input_path}")
    print(f"op: {save_path}")
    print(f"segment: {args.segment}")
    print(f"foundation model: {args.foundation_model}")
    if args.get_features:
        print("getting features from hugging face")


    # 1. read in outcomes
    outcome_df = pd.read_csv(os.path.join(args.truth_path,args.truth_file))
    outcome_df = outcome_df.set_index('ROI_ID')
    print(outcome_df.head())    
    sig_name = outcome_df.columns[0]

    # 2. get features
    GET_FEATURES = args.get_features
    model_name = args.foundation_model

    if GET_FEATURES:
        # load the image patches
        patch_paths = glob.glob(os.path.join(args.input_path,'patches',segment,'*.png'))
        image_names = [os.path.basename(path) for path in patch_paths]
        patch_metadata = pd.DataFrame([x.replace('.png','').split('_') for x in image_names],columns=['name', 'LEAP_ID', 'Segment','ROI_num', 'x', 'y'])
        patches = [Image.open(path) for path in patch_paths]

        # login to hugging face
        login(token="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", add_to_git_credential=False)

        # get features from hugging face
        feature_matrix = extractFeatures(model_name, patches)

        # save the features
        output_path = os.path.join(root_path,'features')
        save_features_with_names(feature_matrix, patch_paths, output_path, f"{model_name}-{segment}", format="csv")

    else:
        if args.feature_file:
            feat_file = args.feature_file
        else:
            feat_file= f'{model_name}-{segment}_features.csv'
        image_names, feature_matrix = load_features_from_csv(os.path.join(args.input_path,feat_file))

        """if i'm reading features from a file then have to get metadata from image names"""
        patch_metadata = pd.DataFrame([x.replace('.png','').split('_') for x in image_names],columns=['name', 'LEAP_ID', 'Segment','ROI_num', 'x', 'y'])

    # remove patches that we don't have both features and outcomes
    patch_metadata = patch_metadata.set_index('name')
    merged_df = patch_metadata.join(outcome_df, how='left')
    mask = ~merged_df[sig_name].isna()
    feature_matrix_clean = feature_matrix[mask.values]
    merged_df = merged_df.dropna(subset=[sig_name])

    set_seed()

    print("Feature Matrix Shape:", feature_matrix_clean.shape) 
    print("Outcome Shape:", len(merged_df))  # Should match the number of images


    print("about to train")

    # TRAIN
    trained_models, scores = train(feature_matrix, merged_df, outcome_df.columns)

    print("BEST PARAMS:")
    for label, model in trained_models.items():
        print(f"{label}: {model.best_params_}")

    print("now concat scores")
    # EVALUATE
    allscores = pd.DataFrame({
        'Run': pd.Series(dtype='str'),
        'MSE': pd.Series(dtype='float'),
        'RMSE': pd.Series(dtype='float'),
        'MAE': pd.Series(dtype='float'),
        'R2': pd.Series(dtype='float'),
        'Pearson Corr': pd.Series(dtype='float'),
        'Spearman Corr': pd.Series(dtype='float'),
        'Train CV Score': pd.Series(dtype='float'),
        'feature_extractor': pd.Series(dtype='str')
    })
    allscores = pd.concat([allscores, scores], ignore_index=True)
    print(allscores.sort_values(by='Spearman Corr', ascending=False))
    print("FINISHED RUNNING")
    
