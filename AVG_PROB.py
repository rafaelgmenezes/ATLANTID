# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 14:19:36 2024

@author: Usuário
"""
# IMPORT PACKAGES 
import os
import pandas as pd


#%% AVG PROBS KFOLD_5

model_path_kf = r'D:\\WHISTLE_CLASSIFIER_FINAL\\8sp\\KFOLD\\KFOLD_PROBS_AVG'
#path for class probs files
os.chdir(model_path_kf)

files = os.listdir()

# Read each CSV file, selecting only the first 9 columns, and store them in a list
first_nine_cols = [pd.read_csv(file).iloc[:, :7] for file in files]

# Calculate the mean across the first nine columns for each row
average_first_nine = pd.concat(first_nine_cols).groupby(level=0).mean()

# Read the first file again to get all columns (including those after the 9th column)
# and replace the first 9 columns with the calculated averages
Kfold_mf = pd.read_csv(files[0])
Kfold_mf.iloc[:, :7] = average_first_nine

#path to save average probs (meta-features) 
#os.chdir(f'D:\WHISTLE_CLASSIFIER_FINAL\{meta_feat_path}')

# Save the resulting DataFrame with averaged values for the first 9 columns to a new CSV file
#base_df.to_csv(f'CLASS_PROBS_AVG_{meta_feat_name}.csv', index=False)



#%% AVG PROBS LOGO_WH

model_path_logo_WH = r'D:\WHISTLE_CLASSIFIER_FINAL\8sp\LOGO\LOGO_PROBS_AVG_WH'

#meta_feat_path = ''

#meta_feat_name = ''

#path for class probs files
os.chdir(model_path_logo_WH)

files = os.listdir()
files

# Read each CSV file, selecting only the first 9 columns, and store them in a list
first_nine_cols = [pd.read_csv(file).iloc[:, :7] for file in files]

# Calculate the mean across the first nine columns for each row
average_first_nine = pd.concat(first_nine_cols).groupby(level=0).mean()

# Read the first file again to get all columns (including those after the 9th column)
# and replace the first 9 columns with the calculated averages
logo_WH_mf = pd.read_csv(files[0])
logo_WH_mf.iloc[:, :7] = average_first_nine




#%% AVG PROBS LOGO_WE

model_path_logo_WE = r'D:\WHISTLE_CLASSIFIER_FINAL\8sp\LOGO\LOGO_PROBS_AVG_WE'

#meta_feat_path = ''

#meta_feat_name = ''

#path for class probs files
os.chdir(model_path_logo_WE)

files = os.listdir()
files

# Read each CSV file, selecting only the first 9 columns, and store them in a list
first_nine_cols = [pd.read_csv(file).iloc[:, :7] for file in files]

# Calculate the mean across the first nine columns for each row
average_first_nine = pd.concat(first_nine_cols).groupby(level=0).mean()

# Read the first file again to get all columns (including those after the 9th column)
# and replace the first 9 columns with the calculated averages
logo_WE_mf = pd.read_csv(files[0])
logo_WE_mf.iloc[:, :7] = average_first_nine

# Save the resulting DataFrame with averaged values for the first 9 columns to a new CSV file
#base_df.to_csv("CLASS_PROBS_AVG_LOGO_WE.csv", index=False)

###############################################################################

#%% CONCATENATE AVG PROBS AND CREATE META-FEATURES SHEET 

meta_feat_path = r'D:\WHISTLE_CLASSIFIER_FINAL\8sp\Meta_features_8sp'

meta_feat_name = '_8sp'

#path for class probs files
os.chdir(meta_feat_path)

os.listdir()

# Kf_5 = pd.read_csv('CLASS_PROBS_AVG_KFOLD_5.csv')
# Lg_wh = pd.read_csv('CLASS_PROBS_AVG_LOGO_WH.csv')
# Lg_we = pd.read_csv('CLASS_PROBS_AVG_LOGO_WE.csv')



# Collect all UIDs from the DataFrames
uids_Kf_5 = set(Kfold_mf['UID'])
uids_Lg_wh = set(logo_WH_mf['UID'])
uids_Lg_we = set(logo_WE_mf['UID'])

# Find UIDs that are unique to each DataFrame
unique_uids_Kf_5 = uids_Kf_5 - uids_Lg_wh - uids_Lg_we
unique_uids_Lg_wh = uids_Lg_wh - uids_Kf_5 - uids_Lg_we
unique_uids_Lg_we = uids_Lg_we - uids_Kf_5 - uids_Lg_wh

# Warn if there are UIDs not matching in all DataFrames
if unique_uids_Kf_5 or unique_uids_Lg_wh or unique_uids_Lg_we:
    print("Warning: Some UIDs do not match across DataFrames!")
    if unique_uids_Kf_5:
        print(f"UIDs unique to Kf_5: {unique_uids_Kf_5}")
    if unique_uids_Lg_wh:
        print(f"UIDs unique to Lg_wh: {unique_uids_Lg_wh}")
    if unique_uids_Lg_we:
        print(f"UIDs unique to Lg_we: {unique_uids_Lg_we}")
else:
    
    print('\n\t UIDs present in all files (no unique UIDs)')
    
    
# Merging the DataFrames on 'UID'
merged_df = Kfold_mf.merge(logo_WH_mf, on='UID', how='outer').merge(logo_WE_mf, on='UID', how='outer')

# Display the resulting DataFrame
print(merged_df)

###drop columns
merged_df.columns
columns_to_drop  = ['Species_Classification', 'KnownSpecies_x', 'EncounterNumber_x','KnownSpecies_y', 'EncounterNumber_y' ]

meta_features = merged_df.drop(columns = columns_to_drop)

#move UID to the last column

col_to_move = 'UID'
cols = list(meta_features.columns)

cols.remove(col_to_move)
cols.append(col_to_move)

meta_features = meta_features[cols]

meta_features.info()
meta_features.head()


# Save the merged DataFrame to a CSV file
meta_features.to_csv(f'META_FEATURES{meta_feat_name}.csv', index=False)









