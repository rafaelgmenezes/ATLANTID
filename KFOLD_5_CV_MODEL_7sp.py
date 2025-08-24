


# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 12:12:54 2024

@author: Alexandre Paro
"""


###############################################################################
#                    CLASSIFIER CROSS - VALIDATION
#                  STRATIFIED KFOLD 5 - BALANCE WEIGTH 
#                            

###############################################################################+



#%% IMPORT PACKAGES
import os
import numpy as np
import pandas as pd
import time 
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
#from sklearn.utils import resample

from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV


import matplotlib.pyplot as plt
import seaborn as sns

#import dill
#import time

pd.set_option('display.max_rows', None )
pd.set_option('display.max_columns', None )
pd.set_option('display.width', 100)


#%% READ THE MASTER WHISTLE DATA 

os.chdir(r'C:\Users\alebi\Desktop\Tese Desktop LG\1- RESULTADOS TESE 031124 ASSOVIOS\7sp\Master_Whistle_7sp')

# os.chdir(r'F:\BACKUP 12 JAN 25\WHISTLE_CLASSIFIER_FINAL\7sp\Master_Whistle_7sp')

Master_Whistle = pd.read_csv('Master_Whistle_7sp.csv') ##USER DEFINED (File name)
Master_Whistle.info()



#%%#%% Count the number of events and whistles by species 
Current_MW = Master_Whistle
n_events_sp = Current_MW.groupby("KnownSpecies")['EncounterNumber'].nunique()
n_whistle_sp = Current_MW.groupby("KnownSpecies")['EncounterNumber'].count()

n_events_whistle_sp = pd.DataFrame({'n_events': n_events_sp, 'n_whistle': n_whistle_sp})

n_events_total = n_events_whistle_sp['n_events'].sum()

prop_events_sp = round(n_events_sp / n_events_total,2)


prop_whistle_sp = round ((n_whistle_sp / len(Current_MW)) *100, 2)

n_events_whistle_sp = pd.DataFrame({'n_events': n_events_sp, '% events': prop_events_sp, 'n_whistle': n_whistle_sp, 
                                   '% whistle': prop_whistle_sp})

print(n_events_whistle_sp)

n_species = len(Current_MW["KnownSpecies"].unique())
print(f'\n\n\tTotal number of species: {n_species }\n')

print(f'\n\tTotal number of events: {n_events_total}\n\n')


#%% DEFINE MODEL'S NAME

#AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA



os.chdir(r'C:\Users\alebi\Desktop\1- RESULTADOS TESE 031124\7sp\KFOLD\KFOLD_E')
os.listdir()
fold_path = r'C:\Users\alebi\Desktop\1- RESULTADOS TESE 031124\7sp\KFOLD\KFOLD_E'
#AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA


#%% DEFINE RANDOM SATES 

#%%########## A ##################

model = 'KFOLD_7sp_A' 


RS_bal_1 = 56450
RS_bal_2 = 6273
RS_bal_3 = 7833

RS_skf = 31853
RS_rf = 13444

n_estimators = 200
max_features = 'log2'
max_depth = 20

#%%########## B ##################

# model = 'KFOLD_7sp_B' 


# RS_bal_1 = 97645
# RS_bal_2 = 86067
# RS_bal_3 = 62224

# RS_skf = 76360
# RS_rf = 92667

# n_estimators = 500
# max_features = 'log2'
# max_depth = 10

#%%########## C ##################

# model = 'KFOLD_7sp_C' 

# RS_bal_1 = 63454
# RS_bal_2 = 2116
# RS_bal_3 = 79859

# RS_skf = 95598
# RS_rf = 34601

# n_estimators = 1000
# max_features = 'log2'
# max_depth = 15


#%%########## D ##################

# model = 'KFOLD_7sp_D' 

# RS_bal_1 = 9771
# RS_bal_2 = 36711
# RS_bal_3 = 25882

# RS_skf = 82486
# RS_rf = 51901

# n_estimators = 200
# max_features = 10
# max_depth = 15

#%%########## E ##################

# model = 'KFOLD_7sp_E' 

# RS_bal_1 = 50978
# RS_bal_2 = 58443
# RS_bal_3 = 13185

# RS_skf = 23093

# RS_rf = 25793

# n_estimators = 1000
# max_features = 0.75
# max_depth = 15


#%% FUNCTION FOR BALANCING \ WEIGTHS - Fix a random state
 
def balance_train_data_kfold(X, weights, target_class_size):
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

#%%Define the predictors and target from train bal 

X = Master_Whistle.drop(['Source', 'EncounterID','CruiseID', 'UID' ], axis=1) #KnoSp e EncNum Mantido!!!
X.info()                                                                       #'OriginalSpecies'
y = Master_Whistle.loc[:, 'KnownSpecies']
y.shape


#%% RUN A KFOLD TO TRAIN AND TEST MODEL AND GET PROBABILITIES - BALACING DATA \WEIGHTS


 # Set your random state for reproducibility

rf_model = RandomForestClassifier(n_estimators = n_estimators, oob_score = True, max_features= max_features, 
                                  max_depth = max_depth,  min_samples_leaf = 1,  min_samples_split= 2, 
                                  random_state= RS_rf) #4 #^^^^^^^^^^ RF PARAMS 
                                  
                                  #^^^^^^^^^^^^^^^^^^^^ RS for RF 

# Prepare to collect out-of-fold probabilities
oof_probabilities = np.zeros((X.shape[0], len(np.unique(y))))

#List for OOB score
OOBs_score = []

All_metrics_report = []

Macro_metrics = pd.DataFrame(columns=['Accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 
                                      'f1_weigthed_avg', 'Balanced_accuracy'])

stats_folds = []

n_samples = len(X)


# Create Stratified K-Fold object
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state= RS_skf) #************************ RS

# Loop over each fold
#for train_index, test_index in skf.split(X, y):
for train_index, test_index in skf.split(X.drop(columns=['EncounterNumber', 'KnownSpecies']), y):
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    
    # Gather statistics for the training set
    train_species_counts = y_train.value_counts()
    total_train = len(X_train)
    n_train_species = len(y_train.unique())
    n_train_groups = len(X_train['EncounterNumber'].unique())

    # Gather statistics for the testing set
    test_species_counts = y_test.value_counts()
    total_test = len(X_test)
    n_test_species = len(y_test.unique())
    n_test_groups = len(X_test['EncounterNumber'].unique())
    
    
    # Calculate weights for balancing
    weights = {}
    for class_label in y_train.unique():
        #class_data = y_train[y_train == class_label]
        # Use EncounterNumber for group calculations
        group_sizes = X_train[X_train['KnownSpecies'] == class_label].groupby('EncounterNumber').size()
        inverse_group_sizes = 1 / group_sizes
        weights[class_label] = (inverse_group_sizes / inverse_group_sizes.sum()).to_dict()
    
    # Determine target class size
    target_class_size = y_train.value_counts().min()

    # Balance the training data within the fold
    X_train_balanced, y_train_balanced = balance_train_data_kfold(X_train, weights, target_class_size)
    
    # Gather statistics for the balanced training set
    train_bal_species_counts = y_train_balanced.value_counts()
    total_train_bal = len(X_train_balanced)
    n_train_bal_species = len(y_train_balanced.unique())
    n_train_bal_groups = len(X_train_balanced['EncounterNumber'].unique())
    
    # Remove 'EncounterNumber' and 'KnownSpecies' from the balanced training data for fitting the model
    X_train_balanced = X_train_balanced.drop(columns=['EncounterNumber', 'KnownSpecies'], errors='ignore')
    # Remove 'EncounterNumber' and 'KnownSpecies' from the test data for predictions
    X_test_features = X_test.drop(columns=['EncounterNumber', 'KnownSpecies'], errors='ignore')
    
    # Train the model on the balanced training fold
    rf_model.fit(X_train_balanced, y_train_balanced)
    
    OOBs_score.append(rf_model.oob_score_)
    
    y_pred = rf_model.predict(X_test_features)
    
    # Get probabilities for the test fold
    oof_probabilities[test_index] = rf_model.predict_proba(X_test_features)
    
    # Get detailed classification report for this fold
    class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # # Extract macro metrics
    accuracy = class_report['accuracy']  # Overall accuracy
    precision_macro = class_report['macro avg']['precision']
    recall_macro = class_report['macro avg']['recall']
    f1_macro = class_report['macro avg']['f1-score']
    f1_weighted = class_report['weighted avg']['f1-score']
    
    bal_accuracy = balanced_accuracy_score(y_test, y_pred)
    
    # Append the report to the list
    All_metrics_report.append(pd.DataFrame(class_report).transpose())
    
    # Store macro metrics for this fold
    Macro_metrics.loc[len(Macro_metrics)] = [accuracy, precision_macro, recall_macro, 
                                             f1_macro, f1_weighted, bal_accuracy]

    # Find labels with no predictions
    labels_no_predictions = []
    for label, metrics in class_report.items():
        if isinstance(metrics, dict) and metrics['precision'] == 0 and test_species_counts.get(label, 0) > 0:
            labels_no_predictions.append(label)
    
    print(f" Labels with no predictions: {labels_no_predictions}")
    
    # Store results for analysis
    stats_folds.append({
        'fold': train_index,
        
        'train_species_counts': train_species_counts,
        'total_train': total_train,
        'n_train_species': n_train_species,
        'n_train_groups': n_train_groups,
        
        'test_species_counts': test_species_counts,
        'total_test': total_test,
        'n_test_species': n_test_species,
        'n_test_groups': n_test_groups,
        
        'train_bal_species_counts': train_bal_species_counts,
        'total_train_bal': total_train_bal,
        'n_train_bal_species': n_train_bal_species,
        'n_train_bal_groups': n_train_bal_groups,
        
        
        'labels_no_predictions': labels_no_predictions,  # Track labels with no predictions
    })
    
    
 # Print statistics for each fold
 
# Print statistics for each fold
for result in stats_folds:
    print(f"Fold {result['fold']}:")
    
    print("\nTraining set statistics:")
    print(result['train_species_counts'])
    print(f'\nTotal training: {result["total_train"]} ({result["total_train"] / n_samples:.2%})')
    print(f'Total number of species: {result["n_train_species"]}')
    print(f'Total number of groups: {result["n_train_groups"]}')
    
    print("\nTesting set statistics:")
    print(result['test_species_counts'])
    print(f'\nTotal testing: {result["total_test"]} ({result["total_test"] / n_samples:.2%})')
    print(f'Total number of species: {result["n_test_species"]}')
    print(f'Total number of groups: {result["n_test_groups"]}\n')    
    
    print("\nTraining balanced set statistics:")
    print(result['train_bal_species_counts'])
    print(f'\nTotal training: {result["total_train_bal"]} ({result["total_train_bal"] / n_samples:.2%})')
    print(f'Total number of species: {result["n_train_bal_species"]}')
    print(f'Total number of groups: {result["n_train_bal_groups"]}\n\n\n')
    print('-' * 8)
    
#%% METRICS


# OOB
OOB_df = pd.DataFrame([OOBs_score], columns= ['1', '2', '3', '4', '5']).T
OOB_df_mean = pd.DataFrame({ 'OOB_mean': OOB_df.mean(), 'OOB_sdv': OOB_df.std()  }) 
              

# Convert each classification report to a DataFrame and calculate the average
df_all_reports = pd.concat(All_metrics_report)
df_avg_metrics = df_all_reports.groupby(df_all_reports.index).mean()

#avg balanced accuracy 
Avg_folds_bal_acc = Macro_metrics['Balanced_accuracy'].mean()

last_column_mean = Macro_metrics.iloc[:, -1].mean()

# Create a new row with None values for all columns
new_row = {col: None for col in Macro_metrics.columns}  # Set None for all columns
new_row[Macro_metrics.columns[-1]] = last_column_mean  

# Create a DataFrame for the new row
new_row_df = pd.DataFrame([new_row])

# Concatenate the new row to the original DataFrame
Macro_metrics = pd.concat([Macro_metrics, new_row_df], ignore_index=True)

# Print results
print(f'\n\tOOB Mean:\n {OOB_df.mean()}\n')
print(f'\n\tMacro metrics for each fold:\n {Macro_metrics}\n')
print(f'\n\tMacro Metrics Mean:\n {df_avg_metrics}')
print(f'\nAverage Balanced Accuracy between folds:\n {round(Avg_folds_bal_acc,2)}\t')

# save csv with results
# OOB_df_mean.to_csv(f'OOB_mean_{model}.csv', index = False)
# Macro_metrics.to_csv(f'Folds_Macro_Metrics_{model}_bal_acc.csv', index = False)
# df_avg_metrics.to_csv(f'Average_Metrics_{model}.csv', index = True)

# Now, `oof_probabilities` will contain the predicted probabilities for all instances in the original dataset.

#%% CLASS PROBS AND OUT OF FOLDS (OOF) RESULTS \ WEIGTHS


#Assuming oof_probabilities is your array with shape (n_samples, n_classes)
# Create a DataFrame from the out-of-fold probabilities
class_labels = rf_model.classes_ 

 # Get the class labels from the trained model
oof_df = pd.DataFrame(oof_probabilities, columns=class_labels)


#Species Classification column 
oof_df['Species_Classification'] = oof_df.idxmax(axis=1)

#include the original indices 
#oof_df['original_index'] = X.index  #to keep track of original indices

oof_df['UID'] = Master_Whistle['UID'].values  
oof_df['KnownSpecies'] = Master_Whistle['KnownSpecies'].values  
oof_df['EncounterNumber'] = Master_Whistle['EncounterNumber'].values 


oof_df.columns = [
    
    col + '_KFOLD_5' if i < 7 else col ###%%%%%%%%%%%%%%% CHANGE accordig to n labels 
    for i, col in enumerate(oof_df.columns)
                                    ]
oof_df.info()

#save class probs
# os.chdir(r'D:\WHISTLE_CLASSIFIER_FINAL\7sp\KFOLD\KFOLD_PROBS_AVG')
# oof_df.to_csv(f'CLASS_PROBS_OOF_{model}.csv', index = False)

#%%## OUT OF FOLDS METRICS \ WEIGTHS
# This metrics represents the agreggate out of folds results (instead of one set of metrics for each fold as previously) 
# os.chdir(fold_path)

oof_df.info()

y_pred_oof = oof_df['Species_Classification']
y_test_oof = oof_df['KnownSpecies']

accuracy =  accuracy_score(y_test_oof, y_pred_oof) 

precision_macro = precision_score(y_test_oof, y_pred_oof, average='macro')
recall_macro = recall_score(y_test_oof, y_pred_oof, average='macro')
f1_macro = f1_score(y_test_oof, y_pred_oof, average='macro')
bal_acc_off = balanced_accuracy_score(y_test_oof, y_pred_oof)

print('\nAccuracy:', accuracy)
print(f'\nBalanced Accuracy OOF: {bal_acc_off:.4f}')
print(f"Macro Precision: {precision_macro:.4f}")
print(f"Macro Recall: {recall_macro:.4f}")
print(f"Macro F1-Score: {f1_macro:.4f}\n\n")


#Save Classifcation Report 

class_report_oof = classification_report(y_test_oof, y_pred_oof, output_dict=True, labels = class_labels )

class_report_oof_df = pd.DataFrame(class_report_oof).transpose()
class_report_oof_df['Labels'] = class_report_oof_df.index

# Reorder columns to have labels first
class_report_oof_df = class_report_oof_df[['Labels', 'precision', 'recall', 'f1-score', 'support']]

# Criar uma nova linha com 'Labels' sendo 'balanced_accuracy' e 'precision' sendo o valor da balanced_accuracy
new_row = pd.DataFrame([['balanced_accuracy', bal_acc_off] + [None] * (class_report_oof_df.shape[1] - 2)],
                       columns=class_report_oof_df.columns)

# Concatenar a nova linha no DataFrame existente
class_report_oof_df = pd.concat([class_report_oof_df, new_row], ignore_index=True)

# class_report_oof_df.to_csv(f'Class_Report_OOF_{model}_bal_acc.csv', index = False)

import dill
# os.chdir(r'C:\Users\alebi\Desktop\1- RESULTADOS TESE 031124\7sp\LOGO\LOGO_B')#
os.chdir(r'C:\Users\alebi\Desktop\FIGURES_SUB_PLOT_WHISTLE_ARTICLE')
os.listdir()

# dill.dump_session('Classifier_whistle_Kfold_A.pkl')
dill.load_session('Classifier_whistle_Kfold_A.pkl')

#%% CONFUSION MATRIX OFF \ WEIGTHS

# Path for paper confusion matrix

os.chdir(r'C:\Users\alebi\Desktop\Tese Desktop LG\1- RESULTADOS TESE 031124 ASSOVIOS\7sp\CM_for_paper')
os.listdir()
       
               
conf_matrix = confusion_matrix(y_test_oof, y_pred_oof, labels = class_labels)

# Compute the confusion matrix percentage
cm_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

# Function to round and adjust the confusion matrix percentages
def round_and_adjust(matrix):
    rounded_matrix = np.floor(matrix * 100).astype(int)  # Round down to ensure totals <= 100
    row_sums = rounded_matrix.sum(axis=1)
    
    for i in range(len(row_sums)):
        errors = 100 - row_sums[i]  # Calculate how much we need to adjust
        if errors > 0:
            # Increment the highest decimal part entries
            decimal_part = (matrix[i] * 100) - np.floor(matrix[i] * 100)
            adjust_indices = np.argsort(decimal_part)[-errors:]  # Get indices with the highest decimals
            rounded_matrix[i, adjust_indices] += 1

    return rounded_matrix

# Adjusted confusion matrix percentages that sum up to 100% in each row
cm_percent_adjusted = round_and_adjust(cm_percent)

# # Calculate diagonal values for sorting
# diagonal_values = np.diag(cm_percent_adjusted)

# # Get sorted indices based on diagonal values (highest to lowest)
# sorted_indices = np.argsort(diagonal_values)[::-1]

# # Create a new confusion matrix that reflects the sorted labels while preserving original percentages
# cm_percent_sorted = cm_percent_adjusted[sorted_indices][:, sorted_indices]

# # Reorder the class labels according to the sorted diagonal values
# class_labels_sorted = [class_labels[i] for i in sorted_indices]

# Convert figure size to inches (1 inch = 25.4mm)
fig_width = 180 / 25.4  # 180mm in inches
fig_height = 5.08  # Height in inches

# Create a heatmap for the confusion matrix
plt.figure(figsize=(fig_width, fig_height))
heatmap = sns.heatmap(cm_percent_adjusted / 100, 
                       annot= True,  #[["{}%".format(int(val)) for val in row] for row in cm_percent_sorted], 
                      cmap="Greys", fmt='',  
                      xticklabels=class_labels,  # Sorted labels for columns
                      yticklabels=class_labels,  # Sorted labels for rows
                      annot_kws={"size": 10}, 
                      cbar_kws={"shrink": 0.8}, 
                      linewidths=0.0, 
                      linecolor='white')

# Customize the spines (margins)
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.3)  # Set spine linewidth

# Access the color bar object and customize
cbar = heatmap.collections[0].colorbar
cbar.outline.set_edgecolor('black')  # Set color of the outline
cbar.outline.set_linewidth(0.2)  # Set outline linewidth

# Customize labels with bold font
plt.xlabel('Predicted Species', fontsize=12, fontweight='bold', labelpad = 10)
plt.ylabel('True Species', fontsize=12, fontweight='bold', labelpad = 10)

plt.xticks(rotation = 45)

plt.show()
# Save the figure
# plt.savefig(f"Conf_Matriz_OOF_{model}_lb_10.png", 
#             format='png', bbox_inches="tight", dpi=300)

#%%
'''
#%% GET A BALANCED FROM THE WHOLE DATASET FOR THE GRID SEARCH CV \ WEIGTHS


Current_MW.info()

X_all = Current_MW  ################################################ UD
X_all.info()
y_all = Current_MW.loc[:, 'KnownSpecies'] ########################## UD
y_all.shape



# Use the same customized function (and same random state)
target_class_size = X_all['KnownSpecies'].value_counts().min()


weights = {}
for class_label in y_all.unique():
    #class_data = y_all[y_all == class_label]
     # Use EncounterNumber for group calculations
    group_sizes = X_all[X_all['KnownSpecies'] == class_label].groupby('EncounterNumber').size()
    inverse_group_sizes = 1 / group_sizes
    weights[class_label] = (inverse_group_sizes / inverse_group_sizes.sum()).to_dict()
 
    
 
X_all_bal, y_all_bal = balance_train_data_kfold(X_all, weights, target_class_size)


X_all_bal.info()

y_all_bal.info()
 
X_all_concat = pd.concat([y_all_bal, X_all_bal], axis = 1)

X_all_concat.info()

#save Train bal data 
X_all_concat.to_csv(f'ALL_DATA_bal_{model}.csv', index = False)


#%% CHECK ALL DATA BALANCED SUMMARY 

n_events_sp = X_all_concat.groupby("KnownSpecies")['EncounterNumber'].nunique()
n_whistle_sp = X_all_concat.groupby("KnownSpecies")['EncounterNumber'].count()

n_events_whistle_sp = pd.DataFrame({'n_events': n_events_sp, 'n_whistle': n_whistle_sp})

n_events_total = n_events_whistle_sp['n_events'].sum()

prop_events_sp = round(n_events_sp / n_events_total,2)


prop_whistle_sp = round ((n_whistle_sp / len(X_all_concat)) *100, 2)

n_events_whistle_sp = pd.DataFrame({'n_events': n_events_sp, '% events': prop_events_sp, 'n_whistle': n_whistle_sp, 
                                   '% whistle': prop_whistle_sp})

print(n_events_whistle_sp)

n_species = len(X_all_concat["KnownSpecies"].unique())
print(f'\n\n\tTotal number of species: {n_species }\n')

print(f'\n\tTotal number of events: {n_events_total}\n')

n_w= len(X_all_concat)
print(f'\n\tTotal number of whistles: {n_w}\n\n')


#%% GRID SEARCH KFOLD

all_data_balanced = X_all_concat 
# CREATE TRAIN DATA X (X_train and y_train) 

##select true class and features of the train data 
X_train = all_data_balanced.loc[:, 'FREQMAX':'STEPDUR']
X_train.info()
X_train.shape

#select target from the train data 
y_train = all_data_balanced.loc[:, 'KnownSpecies']
print(f'y_train shape: {y_train.shape}')


#%% DEFINE RF MODEL, CV METHOD AND PARAM GRID 

#Define Rf model 
rf_classifier = RandomForestClassifier(random_state= RS_rf, n_jobs= -1)

# Define cross-validation method
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state= RS_skf)

#Define grid for search 
param_grid_final = {
    'n_estimators': [100, 200, 300, 500, 1000],
    'max_depth': [None, 5, 10, 15, 20, 25, 30],
    'max_features': ['sqrt', 'log2', 10, 0.5, 0.75]
                    }

#%% DEFINE THE GRID SEARCH CV TO RUN 


#change according to the param_grid to be used
grid_search_Kfold = GridSearchCV(estimator = rf_classifier, param_grid = param_grid_final, 
                                       cv=  skf, 
                                       scoring= ['accuracy', 'f1_macro', 'f1_weighted'],
                                       refit='f1_weighted',  # Will refit the model with the best F1-score
                                       n_jobs=-1, verbose= 1,
                                       return_train_score=False)# default - DO NOT Include training scores in the results

#%% START PROCESSING GRID SEARCH

start_time = time.time()

######Fit the grid search to the data##########
grid_search_Kfold.fit(X_train, y_train)#change according to the param_grid to be used
########################################

end_time = time.time()

elapsed_time = (end_time - start_time)/60
print(f"Elapsed time: {elapsed_time} minutes") 

#%% SAVE RESULTS 


# Get the results of the grid search
results = grid_search_Kfold.cv_results_

# Convert cv_results_ into a DataFrame
results_df = pd.DataFrame(results)


#Filter columns with results 
selected_columns = ['param_max_depth', 'param_max_features', 'param_n_estimators',
                    'mean_test_accuracy', 'std_test_accuracy', 'rank_test_accuracy',
                    'mean_test_f1_macro', 'std_test_f1_macro', 'rank_test_f1_macro',
                    'mean_test_f1_weighted', 'std_test_f1_weighted', 'rank_test_f1_weighted']

results_grid = results_df[selected_columns]

# Save the DataFrame to a CSV file
results_grid.to_csv(f'Grid_Search_{model}.csv', index=False)

print("Grid search results saved")


'''