# -*- coding: utf-8 -*-
"""
Created on Wed May 21 17:05:30 2025

@author: Alexandre Paro
"""
#################################
# SWA Dolphin Whistle Classifier 
#################################

#Predict New Data in Acoustic Events batches ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 
#%% Import packages

import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
#from tkinter import Tk
#from tkinter.filedialog import askopenfilename


#set options to see all columns and rows:   
pd.options.display.max_columns = None 
pd.options.display.max_rows = None 


#%% READ THE WHOLEDATASET TO TRAIN KFOLD 5 MODELS

os.chdir(r'C:\Users\alebi\Desktop\Code_for Whistle_predction\Master_Whistle_7sp')
os.listdir()

Master_Whistle = pd.read_csv('Master_Whistle_7sp.csv') ##USER DEFINED (File name)
Master_Whistle.info()

X = Master_Whistle.drop(['Source', 'EncounterID','CruiseID', 'UID' ], axis=1) #KnoSp e EncNum Mantido!!!
X.info()                                                                       #'OriginalSpecies'
y = Master_Whistle.loc[:, 'KnownSpecies']
y.shape

#%% ---------------------------
# BALANCING FUNCTION for the KFOLD CLASSIFIER 
# ---------------------------
def balance_train_data_kfold(X, weights, target_class_size, RS_bal_1=42, RS_bal_2=43, RS_bal_3=44):
    all_sampled_data = pd.DataFrame()

    # Iterate over each class and their corresponding group weights
    for class_label, group_weights in weights.items():
        class_data = X[X['KnownSpecies'] == class_label]
        total_samples = 0
        group_samples = {}
        
        # First pass: Sample from groups according to their weights
        for group, weight in group_weights.items():
            group_data = class_data[class_data['EncounterNumber'] == group]
            n_samples = int(weight * target_class_size)  # Allocate samples based on group weight
            
            # Adjust n_samples if it exceeds available samples in the group
            n_samples = min(n_samples, len(group_data))
            total_samples += n_samples
            
            if n_samples > 0:
                group_samples[group] = group_data.sample(n=n_samples, replace=False, random_state= RS_bal_1)#*************************** RS
        
        # Combine sampled groups for this class
        sampled_data = pd.concat(group_samples.values())
        
        sampled_indices = sampled_data.index.tolist()  # Track sampled indices

        # If we don't have enough samples for this class, sample more from the remaining class data
        if len(sampled_data) < target_class_size:
            remaining_data = class_data[~class_data.index.isin(sampled_indices)]
            
            if not remaining_data.empty:
                # Calculate the number of additional samples needed
                additional_samples = target_class_size - len(sampled_data)
                if additional_samples > 0:
                    # Sample remaining data to fill the gap, ensuring we don't oversample any group
                    remaining_sampled_data = remaining_data.sample(n=additional_samples, replace=False, random_state= RS_bal_2)# ****** RS
                    sampled_data = pd.concat([sampled_data, remaining_sampled_data])
        
        # Ensure the final sample size matches the target_class_size
        if len(sampled_data) > target_class_size:
            sampled_data = sampled_data.sample(n=target_class_size, replace=False, random_state= RS_bal_3)#***************************** RS
        
        # Accumulate the sampled data across all classes
        all_sampled_data = pd.concat([all_sampled_data, sampled_data])
    
    all_sampled_data = all_sampled_data.reset_index(drop=True)
    
    # Separate features and target for the final balanced dataset
    X_balanced = all_sampled_data.drop(columns=['KnownSpecies'])
    y_balanced = all_sampled_data['KnownSpecies'].reset_index(drop=True)
    
    return X_balanced, y_balanced


#%% FUNCTION TO TRAIN KFOLDS MODELS AND STORE THEM 

# Model configurations A–E
model_configs = {
    'A': dict(RS_bal_1=56450, RS_bal_2=6273, RS_bal_3=7833, RS_skf=31853, RS_rf=13444, 
              n_estimators=200, max_features='log2', max_depth=20),
    'B': dict(RS_bal_1=97645, RS_bal_2=86067, RS_bal_3=62224, RS_skf=76360, RS_rf=92667, 
              n_estimators=500, max_features='log2', max_depth=10),
    'C': dict(RS_bal_1=63454, RS_bal_2=2116, RS_bal_3=79859, RS_skf=95598, RS_rf=34601, 
              n_estimators=1000, max_features='log2', max_depth=15),
    'D': dict(RS_bal_1=9771, RS_bal_2=36711, RS_bal_3=25882, RS_skf=82486, RS_rf=51901, 
              n_estimators=200, max_features=10, max_depth=15),
    'E': dict(RS_bal_1=50978, RS_bal_2=58443, RS_bal_3=13185, RS_skf=23093, RS_rf=25793, 
              n_estimators=1000, max_features=0.75, max_depth=15),
}


def train_kfold_models(X, y, model_configs):
    all_models_kfold = {}  # model_label -> list of 5 RF models

    for model_label, cfg in model_configs.items():
        print(f"Training model {model_label}...")

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg['RS_skf'])
        models = []

        for fold_idx, (train_idx, _) in enumerate(skf.split(X.drop(columns=['EncounterNumber', 'KnownSpecies']), y)):
            print(f"  Fold {fold_idx+1}")
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]

            # Balancing
            weights = {}
            for label in y_train.unique():
                group_sizes = X_train[X_train['KnownSpecies'] == label].groupby('EncounterNumber').size()
                inv_group_sizes = 1 / group_sizes
                weights[label] = (inv_group_sizes / inv_group_sizes.sum()).to_dict()

            target_class_size = y_train.value_counts().min()
            X_train_bal, y_train_bal = balance_train_data_kfold(
                X_train,
                weights=weights,
                target_class_size=target_class_size,
                RS_bal_1=cfg['RS_bal_1'],
                RS_bal_2=cfg['RS_bal_2'],
                RS_bal_3=cfg['RS_bal_3'],
            )

            X_train_bal = X_train_bal.drop(columns=['EncounterNumber', 'KnownSpecies'], errors='ignore')

            rf_model = RandomForestClassifier(
                n_estimators=cfg['n_estimators'],
                oob_score=True,
                max_features=cfg['max_features'],
                max_depth=cfg['max_depth'],
                min_samples_leaf=1,
                min_samples_split=2,
                random_state=cfg['RS_rf']
            )
            rf_model.fit(X_train_bal, y_train_bal)
            models.append(rf_model)

        all_models_kfold[model_label] = models

    return all_models_kfold

##############################################################################
# TRAIN KFOLD MODELS 
##############################################################################

all_models_kfold = train_kfold_models(X, y, model_configs)



#%% FUNCTION TO PREDICT NEW DATA WITH THE KFOLD MODELS

def predict_kfold_probs(new_data_clean, all_models_kfold):
    all_avg_probs = []

    for model_label, models in all_models_kfold.items():
        fold_probs = []

        for rf_model in models:
            X_new_input = new_data_clean.drop(columns=['EncounterNumber', 'KnownSpecies'], errors='ignore')
            probs = rf_model.predict_proba(X_new_input)
            probs_df = pd.DataFrame(probs, columns=rf_model.classes_, index=new_data_clean.index)
            fold_probs.append(probs_df)

        model_avg_probs = sum(fold_probs) / len(fold_probs)
        all_avg_probs.append(model_avg_probs)

    final_avg_probs_df_kfold = sum(all_avg_probs) / len(all_avg_probs)
    final_avg_probs_df_kfold.index = new_data_clean.index
    final_avg_probs_df_kfold.columns = [f'{col}_KFOLD_5' for col in final_avg_probs_df_kfold.columns]
    return final_avg_probs_df_kfold


#avg_probs_kfold = predict_kfold_probs(wh_new_data_clean, all_models_kfold)


#%% Function to train LOGO models 

os.chdir(r'C:\Users\alebi\Desktop\Code_for Whistle_predction\TRAIN_DATA_ALL_MODELS')

os.listdir()


# ==== Paths to your pre-balanced LOGO training data ====
csv_paths = [
    'ALL_DATA_bal_LOGO_7sp_A.csv',
    'ALL_DATA_bal_LOGO_7sp_B.csv',
    'ALL_DATA_bal_LOGO_7sp_C.csv',
    'ALL_DATA_bal_LOGO_7sp_D.csv',
    'ALL_DATA_bal_LOGO_7sp_E.csv'
]

# ==== Model hyperparameters for each ====
n_estimators_list = [500, 1000, 1000, 500, 1000]
max_feat_list = ['log2', 0.5, 'log2', 0.5, 'sqrt']
max_depth_list = [10, 10, 30, 10, 10]
random_state_list = [78978, 52029, 78121, 78113, 72661]



def train_logo_ind_models(csv_paths, n_estimators_list, max_feat_list, max_depth_list, random_state_list):
    models_logo = {}

    for i in range(5):
        df = pd.read_csv(csv_paths[i])
        X_train = df.loc[:, 'FREQMAX':'STEPDUR']
        y_train = df['KnownSpecies']

        model = RandomForestClassifier(
            n_estimators=n_estimators_list[i],
            max_features=max_feat_list[i],
            max_depth=max_depth_list[i],
            random_state=random_state_list[i]
        )
        model.fit(X_train, y_train)
        models_logo[f'rf_model_logo_{chr(65 + i)}'] = model

    return models_logo


##############################################################################
# TRAIN LOGO MODELS 
##############################################################################

all_models_logo = train_logo_ind_models(csv_paths, n_estimators_list, max_feat_list, max_depth_list, random_state_list)



#%% FUNCTION TO PREDICT NEW DATA WITH THE LOGO MODELS

def predict_logo_ind_probs(wh_new_data_clean, all_models_logo):
    X_external = wh_new_data_clean.drop(columns=['KnownSpecies', 'EncounterNumber'], errors='ignore')

    logo_prob_list = []

    for model in all_models_logo.values():
        probas = model.predict_proba(X_external)
        class_names = model.classes_
        proba_df = pd.DataFrame(probas, columns=class_names, index=X_external.index)
        logo_prob_list.append(proba_df)

    avg_probs_logo_ind = sum(logo_prob_list) / len(logo_prob_list)
    avg_probs_logo_ind.columns = [f"{col}_LOGO_WH" for col in avg_probs_logo_ind.columns]
    
    return avg_probs_logo_ind, logo_prob_list, class_names


#avg_probs_logo_ind, logo_prob_list, class_names = predict_logo_ind_probs(wh_new_data_clean, all_models_logo)



#%%  FUNCTION FOR LOGO GROUP PROBS

def compute_logo_group_probs(logo_prob_list, n_estimators_list, class_names, target_index):
    """
    Computes group-level averaged probabilities using model votes.
    """
   
    group_level_prob_list = []

    for group_df, n_estimators in zip(logo_prob_list, n_estimators_list):
        if group_df.empty:
            print("Warning: empty group_df encountered, skipping.")
            continue

        vote_matrix = group_df * n_estimators
        sum_votes = vote_matrix.sum(axis=0)
        total_votes = sum_votes.sum()

        if total_votes == 0:
            print("Warning: total votes = 0, setting all probs to 0")
            group_probs = pd.Series(0.0, index=class_names)
        else:
            group_probs = sum_votes / total_votes
            group_probs = group_probs.reindex(class_names, fill_value=0.0)

        group_level_prob_list.append(group_probs)

    avg_logo_group_probs = sum(group_level_prob_list) / len(group_level_prob_list)

    logo_group_df = pd.DataFrame(
        np.repeat([avg_logo_group_probs.values], len(target_index), axis=0),
        columns=[f"{cls}_LOGO_WE" for cls in avg_logo_group_probs.index],
        index=target_index
    )

    return logo_group_df, avg_logo_group_probs


#logo_group_df, avg_logo_group_probs = compute_logo_group_probs(logo_prob_list, n_estimators_list, class_names, wh_new_data_clean.index)


#%% Function to train LOGO META models 


os.chdir(r'C:\Users\alebi\Desktop\Code_for Whistle_predction\TRAIN_DATA_ALL_MODELS_META')

os.listdir()


# ==== Paths to your pre-balanced LOGO training data ====
csv_paths = [
    'ALL_DATA_bal_LOGO_META_7sp_A.csv',
    'ALL_DATA_bal_LOGO_META_7sp_B.csv',
    'ALL_DATA_bal_LOGO_META_7sp_C.csv',
    'ALL_DATA_bal_LOGO_META_7sp_D.csv',
    'ALL_DATA_bal_LOGO_META_7sp_E.csv'
]

# ==== Model hyperparameters for each ====
n_estimators_list = [500, 100, 1000, 500, 100]
max_feat_list = ['log2', 'log2', 'log2', 'log2','log2']
max_depth_list = [10, 10, 10, 10, 20]
random_state_list = [50057, 35678, 39945, 34400, 73897]


def train_logo_meta_models(csv_paths, n_estimators_list, max_feat_list, max_depth_list, random_state_list):
    """
    Train LOGO meta-classifier models from provided pre-balanced CSVs.
    Returns a dictionary of trained models.
    """
    all_models_meta = {}

    for i in range(len(csv_paths)):
        df = pd.read_csv(csv_paths[i])
        X_train = df.loc[:, 'Delphinus_KFOLD_5':'T_truncatus_LOGO_WE']
        y_train = df['KnownSpecies']

        model = RandomForestClassifier(
            n_estimators=n_estimators_list[i],
            
            max_depth=max_depth_list[i],
            random_state=random_state_list[i]
        )

        model.fit(X_train, y_train)
        all_models_meta[f'rf_model_logo_{chr(65 + i)}'] = model

    return all_models_meta


##############################################################################
# TRAIN LOGO META MODELS 
##############################################################################


all_models_meta = train_logo_meta_models(csv_paths, n_estimators_list, max_feat_list, max_depth_list, random_state_list)


#%% FUNCTION FOR LOGO META GROUP PROBS

def predict_logo_meta_group_probs(meta_features_df, all_models_meta, n_estimators_list, class_names):
    """
    Predict group-level class probabilities using trained meta-classifier models.
    
    Parameters:
        meta_features_df: DataFrame with features (same columns used during training).
        all_models_meta: Dict of trained meta models.
        n_estimators_list: List of n_estimators for each model (for weighting votes).
        class_names: List of all class labels (same order as in training).
    
    Returns:
        avg_logo_group_meta_probs: Series with average group-level probabilities.
    """
    import pandas as pd

    logo_meta_prob_list = []

    for model in all_models_meta.values():
        probas = model.predict_proba(meta_features_df)
        proba_df = pd.DataFrame(probas, columns=model.classes_, index=meta_features_df.index)
        logo_meta_prob_list.append(proba_df)

    # Compute weighted vote aggregation
    group_level_meta_prob_list = []

    for group_df, n_estimators in zip(logo_meta_prob_list, n_estimators_list):
        if group_df.empty:
            print("Warning: empty group_df encountered, skipping.")
            continue

        vote_matrix = group_df * n_estimators
        sum_votes = vote_matrix.sum(axis=0)
        total_votes = sum_votes.sum()

        if total_votes == 0:
            print("Warning: total votes = 0, setting all probs to 0")
            group_probs = pd.Series(0.0, index=class_names)
        else:
            group_probs = sum_votes / total_votes
            group_probs = group_probs.reindex(class_names, fill_value=0.0)

        group_level_meta_prob_list.append(group_probs)

    avg_logo_group_meta_probs = sum(group_level_meta_prob_list) / len(group_level_meta_prob_list)
    avg_logo_group_meta_probs.name = "Avg_meta_group_probs"

    return avg_logo_group_meta_probs


#avg_logo_group_meta_probs = predict_logo_meta_group_probs(meta_features_df, all_models_meta, n_estimators_list, class_names)

#%% PREDICT ACOUSTICS EVENTS IN A BATCH 

###Read the new data for prediction

os.chdir(r'C:\Users\alebi\Desktop\Code_for Whistle_predction\new_data_3')
os.listdir()


# Path to folder with new group CSVs
folder_path = r'C:\Users\alebi\Desktop\Code_for Whistle_predction\new_data_3'

results_output_folder = "Classification_results"
os.makedirs(results_output_folder, exist_ok=True)

# filename= 'S_attenuata_PMC_18_A29_RoccaContourStats.csv'
# Store predictions
all_group_predictions = []

logo_probs = []

meta_probs = []

# Loop through all CSV files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        
        print(f"\n\tProcessing Acoustic Event: {filename}")
        
        # Load new group data
        wh_new_data = pd.read_csv(filename)

        col_exclude = (

                      list(wh_new_data.loc[:, 'Source':'EncounterCount'].columns) +
                      list(wh_new_data.loc[:, 'SamplingRate':'DataProvider'].columns) + 
                                      ['GeographicLocation','ClassifiedSpecies', 'DCMEAN', 'DCSTDDEV'] +
                      list(wh_new_data.loc[:, 'DCQUARTER1MEAN':'DCQUARTER4MEAN'].columns) +
                      list(wh_new_data.loc[:, 'FREQBEGSWEEP':'FREQENDDWN'].columns) +
                      list(wh_new_data.loc[:, 'FREQPEAK':'Species_List'].columns)                                                         
                           
                      )

#clean columns 
        wh_new_data_clean = wh_new_data.drop(columns = col_exclude)


        # ======================
        # BASE CLASSIFIER 1: K-FOLD
        
        avg_probs_kfold = predict_kfold_probs(wh_new_data_clean, all_models_kfold)
        
        # BASE CLASSIFIER 2: LOGO individual
        avg_probs_logo_ind, logo_prob_list, class_names = predict_logo_ind_probs(wh_new_data_clean, all_models_logo)

        # BASE CLASSIFIER 3: LOGO group
        probs_logo_group, avg_logo_group_probs  = compute_logo_group_probs(logo_prob_list, n_estimators_list, class_names, wh_new_data_clean.index)


        # ======================
        # Build meta-features
        meta_features_df = pd.concat(
            [avg_probs_kfold, avg_probs_logo_ind, probs_logo_group],
            axis=1
        )

        #meta_features_df.info()

        #The assert statement in Python is used to test whether a condition is True.
        # If it’s True, nothing happens and the program continues. If it’s False, Python raises an AssertionError and stops execution
        assert (avg_probs_kfold.index == avg_probs_logo_ind.index).all()
        assert (avg_probs_kfold.index == probs_logo_group.index).all()


        # ======================
        # Predict with meta-classifier
        avg_logo_group_meta_probs = predict_logo_meta_group_probs(meta_features_df, all_models_meta, n_estimators_list, class_names)
        

        
        # Save the Kfoldprobs 
            
        avg_probs_kfold['Max_Score'] = avg_probs_kfold.max(axis=1)
        avg_probs_kfold['Classification'] = avg_probs_kfold.idxmax(axis=1)
        avg_probs_kfold.to_csv(os.path.join(results_output_folder, f'{filename}_Kfold.csv'))


        # Save the LOGO ind probs 
            
        avg_probs_logo_ind['Max_Score'] = avg_probs_logo_ind.max(axis=1)
        avg_probs_logo_ind['Classification'] = avg_probs_logo_ind.idxmax(axis=1)
        avg_probs_logo_ind.to_csv(os.path.join(results_output_folder, f'{filename}_LOGO_ind.csv'))







        # ======================
        # Classification Decisions

        # LOGO group
        max_score_logo = avg_logo_group_probs.max()
        max_index_logo = avg_logo_group_probs.idxmax()

        # META
        max_score_meta = avg_logo_group_meta_probs.max()
        max_index_meta = avg_logo_group_meta_probs.idxmax()

        # Final decision logic
        if max_index_logo == max_index_meta:
            final_class = max_index_meta
        elif max_index_meta == 'T_truncatus':
            final_class = max_index_meta
        else:
            final_class = max_index_logo

        # ======================
        # Print summary
        print('-------------------------------------------------------------------------------------------')
        species = wh_new_data_clean['KnownSpecies'].unique()[0]
    
        print(f'\nFile: {filename}')
        print(f'\nNumber of whistles: {len(wh_new_data)}\n')


        #print(avg_logo_group_probs)
        
        print(f"Species classification for the LOGO Classifier: \n\n\t {max_index_logo}\n\t Score: {round(max_score_logo, 2)}\n")

        #print(avg_logo_group_meta_probs)
      
        print(f"Species classifcation for the Meta-Classifier: \n\n\t {max_index_meta}\n\t Score: {round(max_score_meta, 2)}\n")
        print(f'\t\nTrue Species: {species}')
        print(f"Final Classification: {final_class}")
        
        print('-------------------------------------------------------------------------------------------')
        # ======================
        # Save results
        result_row = {
            'Species': species,
            "File": filename,
            "N_whistles": len(wh_new_data),
            "LOGO_Score": round(max_score_logo, 2),
            "LOGO_Classification": max_index_logo,
            "Meta_Score": round(max_score_meta, 2),
            "Meta_Classification": max_index_meta,
            "Final_Classification": final_class
                      }        
        
        
        all_group_predictions.append(result_row)
        
        
        logo_dict = {
        'Species': species,
        'File': filename,
        'N_whistles': len(wh_new_data),
        **avg_logo_group_probs.to_dict()# ** dictionary unpacking key-value to combine to other keys
                    }      
        logo_probs.append(logo_dict)
        
        
        meta_dict = {          
            'Species': species,
            'File': filename,
            'N_whistles': len(wh_new_data),
            **avg_logo_group_meta_probs.to_dict()
                    }
        
        meta_probs.append(meta_dict)
        
print(f'\nTask finished. {len(os.listdir(folder_path))-1} Acoustic Events processed')    


#%% Add info in the classification sheets results 


# Add columns to the all group predictions df
         
classification_results_df = pd.DataFrame(all_group_predictions)


classification_results_df['Prediction'] = classification_results_df['Species'] == classification_results_df['Final_Classification']


classification_results_df.to_csv(os.path.join(results_output_folder, "ALL_GROUP_CLASSIFICATION_SUMMARY.csv"), index=False)         
   
        

#Include coluns with results from logo
 
logo_results_df = pd.DataFrame(logo_probs)

# Step 1: Round columns 3 to 9 (Python is 0-based index, so columns 3:9 = index 2 to 8)
cols_to_process = logo_results_df.columns[3:10]
logo_results_df[cols_to_process] = logo_results_df[cols_to_process].round(3)

# Step 2: Find max value in each row for those columns
logo_results_df['Max_Score'] = logo_results_df[cols_to_process].max(axis=1)

# Step 3: Check for ambiguity (more than one max)
logo_results_df['Ambiguity'] = logo_results_df[cols_to_process].eq(logo_results_df['Max_Score'], axis=0).sum(axis=1).gt(1).map({True: 'yes', False: 'no'})

# Step 4: Find second max value
def second_largest(row):
    unique_vals = sorted(set(row), reverse=True)
    return unique_vals[1] if len(unique_vals) > 1 else unique_vals[0]

logo_results_df['Second_Max_Value'] = logo_results_df[cols_to_process].apply(second_largest, axis=1)

# Step 5: Get column name of max and second max
def get_max_col(row):
    max_val = row.max()
    return row[row == max_val].index.tolist()[0]  # first max if tie

def get_second_max_col(row):
    sorted_vals = row.sort_values(ascending=False)
    if sorted_vals.iloc[0] == sorted_vals.iloc[1]:
        return sorted_vals[sorted_vals == sorted_vals.iloc[0]].index.tolist()[1]  # second of the tied maxes
    else:
        return sorted_vals.index[1]

logo_results_df['Max_Col'] = logo_results_df[cols_to_process].apply(get_max_col, axis=1)
logo_results_df['Second_Max_Col'] = logo_results_df[cols_to_process].apply(get_second_max_col, axis=1)

logo_results_df['Prediction'] = logo_results_df['Species'] == logo_results_df['Max_Col']

logo_results_df.to_csv(os.path.join(results_output_folder, "LOGO_CLASSIFICATION_SUMMARY.csv"), index=False) 

#Include coluns with results from meta
meta_results_df = pd.DataFrame(meta_probs)
 
# Step 1: Round columns 3 to 9 (Python is 0-based index, so columns 3:9 = index 2 to 8)
cols_to_process = meta_results_df.columns[3:10]
meta_results_df[cols_to_process] = meta_results_df[cols_to_process].round(3)

# Step 2: Find max value in each row for those columns
meta_results_df['Max_Score'] = meta_results_df[cols_to_process].max(axis=1)

# Step 3: Check for ambiguity (more than one max)
meta_results_df['Ambiguity'] = meta_results_df[cols_to_process].eq(meta_results_df['Max_Score'], axis=0).sum(axis=1).gt(1).map({True: 'yes', False: 'no'})

# Step 4: Find second max value
def second_largest(row):
    unique_vals = sorted(set(row), reverse=True)
    return unique_vals[1] if len(unique_vals) > 1 else unique_vals[0]

meta_results_df['Second_Max_Value'] = meta_results_df[cols_to_process].apply(second_largest, axis=1)

# Step 5: Get column name of max and second max
def get_max_col(row):
    max_val = row.max()
    return row[row == max_val].index.tolist()[0]  # first max if tie

def get_second_max_col(row):
    sorted_vals = row.sort_values(ascending=False)
    if sorted_vals.iloc[0] == sorted_vals.iloc[1]:
        return sorted_vals[sorted_vals == sorted_vals.iloc[0]].index.tolist()[1]  # second of the tied maxes
    else:
        return sorted_vals.index[1]

meta_results_df['Max_Col'] = meta_results_df[cols_to_process].apply(get_max_col, axis=1)
meta_results_df['Second_Max_Col'] = meta_results_df[cols_to_process].apply(get_second_max_col, axis=1)

meta_results_df['Prediction'] = meta_results_df['Species'] == meta_results_df['Max_Col']


meta_results_df.to_csv(os.path.join(results_output_folder, "META_CLASSIFICATION_SUMMARY.csv"), index=False) 





       

        

