# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 15:40:28 2024

@author: Usuário
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 22:26:59 2023

# -*- coding: utf-8 -*-
"""

#@author: Alexandre Paro

##               ~~~~~~~~  LOGO   ~~~~~~~~
################# LEAVE ONE GROUP OUT #########################################
#AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
######                                                              ###############################
#####        SOUTHWESTERN ATLANTIC DOLPHIN WHISTLE CLASSIFIER       ###############################
######                 CLASSIFIER SWA 2024   TREE VOTES             ###############################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                              #7 species
#%% IMPORT PACKAGES

import os
import glob
import pkg_resources
import pandas as pd
import numpy as np
import time
#import random
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneGroupOut
#from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import balanced_accuracy_score

#check pakcage version
pkg_resources.get_distribution('scikit-learn').version 

#set options to see all columns and rows:   
pd.options.display.max_columns = None 
pd.options.display.max_rows = None 

#%% READ THE MASTER WHISTLE DATA 

#%% CHECK IF MASTER WHISTLE ALREADY EXIST 

os.chdir(r'C:\Users\alebi\Desktop\Tese Desktop LG\1- RESULTADOS TESE 031124 ASSOVIOS\7sp\Master_Whistle_7sp')#***>>>> UD
os.listdir()

Current_MW = pd.read_csv('Master_Whistle_7sp.csv')#***>>>> UD

Current_MW.info()

#%% Count the number of events and whistles by species 

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


#%% CLASSIFICADOR LOGO

###############################################################################
#check path and dir content again 
print()
print(os.getcwd())
os.listdir()
glob.glob('*csv')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# DEFINE RANDOM STATES
###############################################################################
#%%########## A ##################

# model = 'LOGO_7sp_A'

# RS_bal = 26225
# RS_rf = 78978

# n_estimators = 500
# max_features= 'log2'
# max_depth= 10

#%%########## B ##################

model = 'LOGO_7sp_B'

RS_bal = 64386
RS_rf = 52029

n_estimators = 1000
max_features= 0.5
max_depth= 10

#%%########## C ##################

# model = 'LOGO_7sp_C'

# RS_bal = 61999
# RS_rf = 78121

# n_estimators = 1000
# max_features= 'log2'
# max_depth= 30

#%%########## D ##################

# model = 'LOGO_7sp_D'

# RS_bal = 71437
# RS_rf = 78113

# n_estimators = 500
# max_features= 0.5
# max_depth= 10

#%%########## E ##################

# model = 'LOGO_7sp_E'

# RS_bal = 1972
# RS_rf = 72661

# n_estimators = 1000
# max_features= 'sqrt'
# max_depth= 10

#%% Definir parametros do classificador  


class Classifier():
    __description__ = ''
    __authors__     = ''
    __date__        = ''

    def __init__(self, dict={}, file = ''):
        self.sep = '='*80
        self.globals={'group':           'golfinhos',
                      'mastersheet_ext': 'csv',
                      'factors':         ['KnownSpecies','EncounterNumber']
                      }
        
        self.balancing = {}

        self.read_mastershet()
        print(self.sep)
        self.summarize_data()
        
        #self.training()

        self.testing_loop()



    def info(self): 
        print('')

    def setpath(self):
        group = self.globals['group']
        if group.lower() in ['golfinhos','peixes']: 
            if os.path.basename(os.getcwd()) == 'Detector': os.chdir('Biofonia/'+group)
        else: print('error. not folder for this group') #raise error

    def read_mastershet(self):
        from glob import glob
        current_path = os.getcwd()
        self.setpath()
        ext =  self.globals['mastersheet_ext']
        file = glob('*.%s'%(ext))
        if len(file) > 1: print("More than 1 '%s' file was found.\nSelecting the first file of the list: %s" %(ext, file)) # add input/tk here
        f = file[0] 
        print(self.sep)
        print('Reading mastersheet...\n\t%s' % (f))
        mastersheet = pd.read_csv(f, engine = 'python')
        self.mastersheet = mastersheet
        os.chdir(current_path)

    def summarize_data(self, df=pd.DataFrame(), cross_factor = True):
        factors = self.globals['factors']
        if df.empty: 
            print('\n%s\nMastersheet summary:' %(self.sep))
            df = self.mastersheet
        else: print('\n%s\nDataFrame summary:'%(self.sep))
        print('\tShape dimensions:  %s'%(str(df.shape)))

        if type(factors) == 'str': factors = [factors]
        for fac in factors:
            print('\t%s:  %s'%(fac, len(df.groupby(fac).groups)))

        if cross_factor:
            for i in range(0,len(factors)-1):
                n_fac2   = df.groupby([factors[i], factors[i+1]]).apply(lambda x: len(x), include_groups = False)
                n_fac1   = pd.Series(df.groupby(factors[i])[factors[i+1]].nunique())
                n_id     = pd.Series(df.groupby(factors[i])[factors[i+1]].count(), name = 'n')
                df_nfac1 = pd.DataFrame([n_fac1,n_id]).T
                print('\n\t%s X %s:  \n%s\n\n%s\n' % (factors[i], factors[i+1], df_nfac1, n_fac2))
                self.summary_data = {'n_fac1': df_nfac1, 'n_fac2': n_fac2}

    def find_minimum (self, df=pd.DataFrame(), print_details = False):
        factor = self.globals['factors']
        if type(factor) == str: factor1 = factor
        elif type(factor) == list: factor1 = factor[0]
        if df.empty: df = self.mastersheet

        id = df.groupby(factor1).apply(lambda x:len(x), include_groups =False).argmin()
        nbal         = df.groupby(factor1).apply(lambda x:len(x), include_groups = False).min()
        factor1_list = df[factor1].drop_duplicates().reset_index(drop=True)

        keymin = list(df.groupby(factor1).groups.keys())[id]
        if print_details:
            print(self.sep)
            print('Key of %s with minimum entries -> %s ' %(factor1, keymin))
            if len (factor) == 2:
                factor2 = factor[1]
                print(df[df[factor1]==keymin].groupby(factor2).apply(lambda x:len(x)))
            print('%s'%(self.sep))
        return nbal, factor1_list
    
    def subsampling(self, df2subsample=pd.DataFrame(), nbal=1, factor_val = '', random_val = RS_bal): ##******* RS - sampling 
        if df2subsample.empty: df2subsample = self.mastersheet
        factors = self.globals['factors']
        factor1 = factors[0]
        factor2 = factors[1]

        if type(factor_val) == str:
            if factor_val == '': factor_val = 0
            else:                value = factor_val
        if type(factor_val) in [float,int]: 
            factor1_list = self.balancing['factor1_list'] 
            #factor1_list = mastersheet[factor1].drop_duplicates().reset_index(drop=True)
            value = factor1_list[factor_val]

        #print('\tsubsampling...\t%s [r = %s]' %(value, random_val))
        filtered_ms = df2subsample[df2subsample[factor1] == value]
        weights = pd.Series(filtered_ms.groupby(factor2)[factor1].transform('count'))
        weights1 = 1/weights
        weights2 = weights1/weights1.sum() #peso normalizad (inserido 16/10/24)
        
        #weights2 = 1/(weights/(weights.max()/2))

        subsample = filtered_ms.sample(n=nbal, weights = weights2, random_state=random_val)#.groupby([factor1,factor2])
        #print(subsample.apply(lambda x:len(x)))
        #len(n_subsample)
        return subsample
        # np.random.seed(seed)

    def training(self, data2train, print_summary = False):
        nbal, factor1_list = self.find_minimum(df=data2train)
        print ('\t\t\tNBAL = %s'%(nbal))
        # loop to create the balanced train data dataframe 
        df_trained = pd.DataFrame()
        for ft in factor1_list:
            subsample = self.subsampling(df2subsample = data2train, nbal = nbal, factor_val = ft)
            df_trained  = pd.concat([df_trained,subsample],axis=0)
        if print_summary: self.summarize_data(df=df_trained)
        #df_train = self.clear_columns(df=df_train)
        return df_trained

    def testing_loop(self):
        
        print('%s\n%s\nInitializing RFC tests in loop...\n%s'%(self.sep,self.sep,self.sep))
        #from numpy import concatenate
        mastersheet = self.mastersheet
        #mastersheet = self.clear_columns(df=mastersheet)
        factors = self.globals['factors']
        factor1 = factors[0]
        factor2 = factors[1]
        factor1_list = mastersheet[factor1].drop_duplicates().reset_index(drop=True)
        results = {}
        for ft1 in factor1_list:
            print('\n%s...'%(ft1))
            df_ft1  = mastersheet[mastersheet[factor1]==ft1]
            ft2_ft1 = df_ft1[factor2].drop_duplicates().reset_index(drop=True)
            #results[ft1][factor2] = ft2_ft1.T.tolist()
            ft2_indexes = []
            ft2_classification = []
            ft2_tree_votes = []
            ft2_class_prob = []
            ft2_OOB_score = []
            for ft2 in ft2_ft1:#[:2]:
                #######################################
                print('\t%s:\n\t\tTraining model...'%(ft2))
                df2train = mastersheet[~(mastersheet[factor2] == ft2)]
                #print('\nANTES SUBSAMPLING\n')
                #self.summarize_data(df2train)
                df2train = self.training(data2train=df2train)
                #print('\nDEPOIS SUBSAMPLING\n')
                #self.summarize_data(df2train)
                
                #sys.exit(1) 
                
                df2train = self.clear_columns(df=df2train)         
                X_train = df2train.drop(factor1, axis=1)   # drop~delete coluna do fator 1 para inserir no treinamento
                y_train = df2train[factor1]
                RFC = RandomForestClassifier(n_estimators = n_estimators, max_features= max_features, max_depth= max_depth, random_state= RS_rf,##***  Params for RF model
                                             oob_score= True, n_jobs= -1)                                                   ##***  RS for RF model 
                RFC.fit(X_train,y_train)
                
                #######################################
                print('\t\tClassifying...')
                df2run = df_ft1[df_ft1[factor2]==ft2]
                df2run = self.clear_columns(df=df2run).drop(factor1,axis=1)
                y_pred = RFC.predict(df2run)
                
                class_params = RFC.get_params()
                class_labels = RFC.classes_
                OOB_score = RFC.oob_score_
                class_prob = RFC.predict_proba(df2run)
                class_prob_df = pd.DataFrame(class_prob, columns = class_labels)
                total_trees  = RFC.n_estimators
                absolute_votes = (class_prob * total_trees) 
                absolute_votes_class = pd.DataFrame(absolute_votes, columns = class_labels)
                absolute_votes_class_round = absolute_votes_class.round(3)
                
                
                #######################################
                ft2_indexes.append(df2run.index.values.tolist())
                ft2_classification.append(y_pred.tolist())
                ft2_class_prob.append(class_prob_df)
                ft2_tree_votes.append(absolute_votes_class_round)
                ft2_OOB_score.append(OOB_score)
                
            results[ft1] = {factor2:ft2_ft1.tolist(), 'Indexes_raw':ft2_indexes,'Classification':ft2_classification, 'Class_probs': ft2_class_prob, 
                            'Tree_votes': ft2_tree_votes, 'Model_parameters': class_params, 'OOB_score': ft2_OOB_score}
                           
        self.results = results

    def clear_columns(self, df=pd.DataFrame()):
        if df.empty: datasheet = self.mastersheet
        else: datasheet = df.copy()
        dummycols = {'golfinhos': ['Source:CruiseID', 'UID', ],#'OriginalSpecies'
                    'peixes': []}
        
        for col in dummycols[self.globals['group']]: 
            if ':' in col:
                col_edges = col.split(':')
                datasheet.drop(datasheet.loc[:,col_edges[0]:col_edges[1]], inplace=True,axis=1)            
            else: datasheet.drop([col], inplace=True,axis=1)
        return datasheet
 
    
####################
#%% Rodar o classifcador #UD

start_time = time.time() 


Classifier_whistle = Classifier() # 
 

end_time = time.time() 

elapsed_time = round(((end_time - start_time) /60) ,2)
    
print(f'Elapsed time: {elapsed_time} minutes')  #40min - 1h 

#################
#%% Salvar pkl - classificador

import dill
# os.chdir(r'C:\Users\alebi\Desktop\1- RESULTADOS TESE 031124\7sp\LOGO\LOGO_B')#
os.chdir(r'C:\Users\alebi\Desktop\FIGURES_SUB_PLOT_WHISTLE_ARTICLE')
os.listdir()

# dill.dump_session('Classifier_whistle_LOGO_B.pkl')

# dill.load_session('Classifier_whistle_LOGO_B.pkl')

#%%    
#############################################################################
#%% DEFINIR DIRETORIO E CHECAR DADOS DA MASTER_WHISTLE DO CLASSIFICADOR

# os.chdir(r'C:\Users\alebi\Desktop\1- RESULTADOS TESE 031124\7sp\LOGO\LOGO_B')# *********UD
# os.listdir()

# File_path = r'C:\Users\alebi\Desktop\1- RESULTADOS TESE 031124\7sp\LOGO\LOGO_B'


obj_tree_vote = Classifier_whistle #USER DEFINED (nome do classificador) 

obj_tree_vote.summarize_data() 
obj_tree_vote.results.keys()

#%% WHISTLE CLASSIFICATION
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#%% ORGANIZE CLASSIFICATION RESULTS
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

###############################################################################
###Confusion Matrix for whistles       

data_Tt = Classifier_whistle.results['T_truncatus']['Classification'] #list of list (each inner list is the classification results for each encounter)
data_Sb = Classifier_whistle.results['S_bredanensis']['Classification']
data_Dsp = Classifier_whistle.results['Delphinus']['Classification']
data_Sc = Classifier_whistle.results['S_clymene']['Classification']
data_Sf = Classifier_whistle.results['S_frontalis']['Classification']
data_Sl = Classifier_whistle.results['S_longirostris']['Classification']
data_Sa = Classifier_whistle.results['S_attenuata']['Classification']
#data_Gsp = Classifier_whistle.results['Globicephala']['Classification']
# data_Pc = Classifier_whistle.results['P_crassidens']['Classification']
# data_Fa = Classifier_whistle.results['F_attenuata']['Classification']
# #data_Pe = Classifier_whistle_jul_24_8sp.results['P_electra']['Classification']

predicted_Tt = np.array([label for sublist in data_Tt for label in sublist]) #loop to get all class res lists of the species(encs) and put it in a single array
predicted_Sb = np.array([label for sublist in data_Sb for label in sublist])
predicted_Dsp = np.array([label for sublist in data_Dsp for label in sublist])
predicted_Sa = np.array([label for sublist in data_Sa for label in sublist])
predicted_Sc = np.array([label for sublist in data_Sc for label in sublist])
predicted_Sf = np.array([label for sublist in data_Sf for label in sublist])
predicted_Sl = np.array([label for sublist in data_Sl for label in sublist])
#predicted_Gsp = np.array([label for sublist in data_Gsp for label in sublist])
# predicted_Pc = np.array([label for sublist in data_Pc for label in sublist])
# predicted_Fa = np.array([label for sublist in data_Fa for label in sublist])
#predicted_Pe = np.array([label for sublist in data_Pe for label in sublist])


class_Tt = ["T_truncatus"]
class_Sb = ["S_bredanensis"]
class_Dsp = ["Delphinus"]
class_Sa = ["S_attenuata"]
class_Sc = ["S_clymene"]
class_Sf = ["S_frontalis"]
class_Sl = ["S_longirostris"]
#class_Gsp = ['Globicephala']
# class_Pc = ['P_crassidens']
# class_Fa = ['F_attenuata']
#class_Pe = ['P_electra']


true_Tt = np.array(class_Tt * len(predicted_Tt)) #array with the trues of the species 
true_Sb = np.array(class_Sb * len(predicted_Sb))
true_Dsp = np.array(class_Dsp * len(predicted_Dsp))
true_Sa = np.array(class_Sa * len(predicted_Sa))
true_Sc = np.array(class_Sc * len(predicted_Sc))
true_Sf = np.array(class_Sf * len(predicted_Sf))
true_Sl = np.array(class_Sl * len(predicted_Sl))
#true_Gsp = np.array(class_Gsp * len(predicted_Gsp))
# true_Pc = np.array(class_Pc * len(predicted_Pc))
# true_Fa = np.array(class_Fa * len(predicted_Fa))
#true_Pe = np.array(class_Pe * len(predicted_Pe))



predict_all = np.hstack((predicted_Dsp, predicted_Sa, predicted_Sb, predicted_Sc, predicted_Sf, predicted_Sl, predicted_Tt))#,predicted_Gsp))
                         #, predicted_Pc, predicted_Fa))#, predicted_Pe)) #horizontal stack to form a single array
                       


True_all = np.hstack(( true_Dsp, true_Sa, true_Sb, true_Sc, true_Sf, true_Sl, true_Tt))#, true_Gsp))#, true_Pc, true_Fa))#, true_Pe))


# Define classes
classes = ["Delphinus", "S_attenuata", "S_bredanensis", "S_clymene", "S_frontalis", "S_longirostris", "T_truncatus"]#,  "Globicephala"] # 'P_electra']
           #"P_crassidens",'F_attenuata'
           
# Classification report 

class_report_wh = classification_report(True_all, predict_all, output_dict=True, labels = classes)
class_report_wh = pd.DataFrame(class_report_wh).transpose()
class_report_wh['Labels'] = class_report_wh.index

# Reorder columns to have labels first
class_report_wh_df = class_report_wh[['Labels', 'precision', 'recall', 'f1-score', 'support']]

print(class_report_wh)

#class_report_wh_df.to_csv(f'Class_report_WH_{model}.csv', index = False)


#Matriz de confusão
cm = confusion_matrix(True_all, predict_all, labels=classes)

# Calculando a especificidade por classe
specificities = []
for i, label in enumerate(classes):
    # VN = Soma de todos os valores, exceto os da linha e coluna da classe atual
    tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
    # FP = Soma da coluna atual, exceto o valor da diagonal principal
    fp = cm[:, i].sum() - cm[i, i]
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    specificities.append(specificity)


# Calculate Macro and Weighted Averages for Specificity
macro_avg_specificity = np.mean(specificities)
weighted_avg_specificity = np.average(specificities, weights=class_report_wh.loc[classes, 'support'])


# Add specificities to DataFrame
specificities.extend([np.nan, macro_avg_specificity, weighted_avg_specificity])  # Add NaN for accuracy row


balanced_acc = balanced_accuracy_score(True_all, predict_all)
print(f"Balanced Accuracy: {balanced_acc:.2f}")


# Adicionando a especificidade ao DataFrame
class_report_wh['specificity'] = specificities

# Reordenar as colunas para incluir a especificidade
class_report_wh_df = class_report_wh[['Labels', 'precision', 'recall', 'f1-score', 'specificity', 'support']]

# Criar uma nova linha com 'Labels' sendo 'balanced_accuracy' e 'precision' sendo o valor da balanced_accuracy
new_row = pd.DataFrame([['balanced_accuracy', balanced_acc] + [None] * (class_report_wh_df.shape[1] - 2)],
                       columns=class_report_wh_df.columns)

# Concatenar a nova linha no DataFrame existente
class_report_wh_df = pd.concat([class_report_wh_df, new_row], ignore_index=True)

# Exibir o DataFrame final
print(class_report_wh_df)

class_report_wh_df.to_csv(f'Class_report_WH_{model}_bal_acc.csv', index = False)


#%% CONFUSION MATRIX - WHISTLES 

#Compute the confusion matrix
cm = confusion_matrix(True_all, predict_all, labels=classes)


class_labels = classes
conf_matrix = confusion_matrix(True_all, predict_all, labels=classes)

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

# Calculate diagonal values for sorting
diagonal_values = np.diag(cm_percent_adjusted)

# Get sorted indices based on diagonal values (highest to lowest)
sorted_indices = np.argsort(diagonal_values)[::-1]

# Create a new confusion matrix that reflects the sorted labels while preserving original percentages
cm_percent_sorted = cm_percent_adjusted[sorted_indices][:, sorted_indices]

# Reorder the class labels according to the sorted diagonal values
class_labels_sorted = [class_labels[i] for i in sorted_indices]

# Convert figure size to inches (1 inch = 25.4mm)
fig_width = 180 / 25.4  # 180mm in inches
fig_height = 5.08  # Height in inches

# Create a heatmap for the confusion matrix
plt.figure(figsize=(fig_width, fig_height))
heatmap = sns.heatmap(cm_percent_sorted / 100, 
                      annot=[["{}%".format(int(val)) for val in row] for row in cm_percent_sorted], 
                      cmap="Greys", fmt='',  
                      xticklabels=class_labels_sorted,  # Sorted labels for columns
                      yticklabels=class_labels_sorted,  # Sorted labels for rows
                      annot_kws={"size": 10}, 
                      cbar_kws={"shrink": 0.8}, 
                      linewidths=0.1, 
                      linecolor='black')

# Customize the spines (margins)
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.1)  # Set spine linewidth

# Access the color bar object and customize
cbar = heatmap.collections[0].colorbar
cbar.outline.set_edgecolor('black')  # Set color of the outline
cbar.outline.set_linewidth(0.1)  # Set outline linewidth

# Customize labels with bold font
plt.xlabel('Espécie Predita', fontsize=12, fontweight='bold')
plt.ylabel('Espécie Real', fontsize=12, fontweight='bold')

# Save the figure
plt.savefig(f'Conf_Matrix_WH_{model}.png', 
            format='png', bbox_inches="tight", dpi=300)


#%%    
#############################################################################

#%% ENCOUNTER CLASSIFICATION   
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



#%% ORGANIZE RESULTS FROM ENCOUNTER CLASSIFCATION





####loop for ecounter classification based in the tree votes 

encs_tree_votes_dict = {}

OOB_score_list = []

Enc_summary_list = []


for species in obj_tree_vote.results.keys():
    
    actual_species = species    
    results_species = obj_tree_vote.results[species]
    Encounters_list = obj_tree_vote.results[species]['EncounterNumber']   
    tree_count_classification =  obj_tree_vote.results[species]['Tree_votes'] #results whistle count
    OOB_score = obj_tree_vote.results[species]['OOB_score']
    OOB_score_list.extend(OOB_score)
    
    

    Enc_summary_df  = pd.DataFrame()
    
    for encounter in tree_count_classification:
        
        # encounter_tree_count = pd.DataFrame(encounter)
        
        # sum_tree = encounter_dataframe.iloc[:, 0: 9].sum()
        
        n_whistle = encounter.shape[0]
        
        sum_tree = encounter.sum()
    
        total_n_tree = sum(sum_tree)
        
        enc_tree_prop = sum_tree/total_n_tree
    
        encounter_probs = pd.concat([encounter, enc_tree_prop.to_frame().T], ignore_index=True)
    
        #object with the enc_propr result (last row)  
        last_row_numeric = encounter_probs.iloc[-1].round(2)
    
        
        Score_classification = float(enc_tree_prop.max())
    
        max_instances = (enc_tree_prop == Score_classification).sum() 
    

        try:
            actual_species_score = round(float(enc_tree_prop.loc[actual_species]), 2)
        
        except:
                actual_species_score = 0.0  
        
     
        if max_instances == 1:
        
            Species_classification = enc_tree_prop.idxmax()
            #Species_classification = Species_classification_series.iloc[0]
            drop_max = enc_tree_prop.drop(enc_tree_prop.idxmax())
        
            if n_whistle > 1:
        
                Score_2nd = float(drop_max.max())
       
                max_instances_2nd_sum = (drop_max == Score_2nd).sum()
                max_instances_2nd = drop_max == Score_2nd #boolean results 
        
                if max_instances_2nd_sum == 1:
                    Species_2nd = max_instances_2nd.index[max_instances_2nd].tolist() # Get the indexes where the values are True
                                                                                   #Coerce it to a list  
                
                else:
                    Species_2nd_list = max_instances_2nd.index[max_instances_2nd].tolist() # Get the indexes where the values are True
                    Species_2nd = ' - '.join(Species_2nd_list) #join species names and create a str object  
    
            else:
                Score_2nd =  None 
                Species_2nd = None 
            
    
        else:
        
            Species_class_more_than_one = enc_tree_prop == Score_classification
            Species_classification_list = Species_class_more_than_one.index[Species_class_more_than_one].tolist()
            Species_classification = ' - '.join(Species_classification_list)
        
            Score_2nd = float(enc_tree_prop.max())
            Species_2nd  = Species_classification
           
    
    
#create a dict with the encounter classifcation results   
    
        dic_enc_summary = {
                       
                       'KnownSpecies': actual_species,
                       'Score_classification': Score_classification,
                       'Species_classification': Species_classification,
                       'Actual_species_score': actual_species_score,
                       'Score_2nd': Score_2nd,
                       'Species_2nd': Species_2nd,
                       'N_whistle': n_whistle
                           
                       }
    
    
#Coerce to a dataframe     
        index = [0]
        df = pd.DataFrame(dic_enc_summary, index = index)
        df_filtered = df.dropna(axis = 1, how = 'all')
#Concatenate the results for all encounters of the species in a single dataframe   
        Enc_summary_df = pd.concat([Enc_summary_df, df_filtered], ignore_index = True)
    
##include a column with the EncounterNumber    
    EncounterNumber  = pd.DataFrame({'EncounterNumber': Encounters_list})
    
    Enc_summary_df['EncounterNumber'] = EncounterNumber
 
       
#Check if species classifcation is correct and a create a series 'True' or 'False'
    Correct = Enc_summary_df['Species_classification'] == Enc_summary_df['KnownSpecies']  

#len(Correct)
#include the 'Correct' column in the dataframe 
    Enc_summary_df['Correct'] = Correct

#Enc_summary_df.shape
#Enc_summary_df.head(5)   
    
    Enc_summary_list.append(Enc_summary_df)
    #Enc_summary_df.to_csv(f'{species}_Class_WE_9sp_LOGO_A.csv', index = False)  
   


# Concatenar os resultadso de classifccao de eventos de assovios das especies 

Enc_Class_results = pd.DataFrame()

for enc in  Enc_summary_list:
    
    df = enc
    
    Enc_Class_results = pd.concat(Enc_summary_list, ignore_index= True)

# Enc_Class_results.to_csv(f'All_species_Class_WE_{model}.csv')



# Enc_Class_results = pd.DataFrame()
# Files = os.listdir()
# len(Files)
# for file in Files:
#     if file.endswith('.csv'):
#         df = pd.read_csv(file, encoding = 'utf-8')
#         Enc_Class_results = pd.concat([Enc_Class_results, df], ignore_index= True)    
    
# #os.chdir(r'D:\WHISTLE_CLASSIFIER_FINAL\META_CLASSIFIER')    
# Enc_Class_results.to_csv(f'All_species_Class_WE_{model}.csv')


print('\n\tWhistle event classifcation results saved succesfully\n')



#%% CHECK IF THERE IS AMBIGUOS ENCOUNTER CLASSIFICATIONS 

print(len(Enc_Class_results['Species_classification'].unique()))
Enc_Class_results['Species_classification'].unique()
##if ambiguos encounters classifications found
# edit no ambiguos mannually edited csv  , choose the 1st in th elist (but not the actual species)
# Enc_Class_results = pd.read_csv('All_species_Class_WE_LOGO_7sp_BF_no_ambiguos.csv') #no_ambiguos

# len(Enc_Class_results['Species_classification'].unique())
# Enc_Class_results['Species_classification'].unique()

#%% METRICS - ENCOUNTER CLASSIFICATION

# # Define classes
# classes = ["Delphinus", "S_attenuata", "S_bredanensis", "S_clymene", "S_frontalis", "S_longirostris", "T_truncatus"]#,  "Globicephala"] # 'P_electra']
#            #"P_crassidens",'F_attenuata'
           
           
# class_labels = sorted(list(classes))
class_labels = ["D_delphis", "S_attenuata", "S_bredanensis", "S_clymene", "S_frontalis", "S_longirostris", "T_truncatus"]#,

# OOB
mean_OOB  = np.mean(OOB_score_list)

sd_OOB = np.std(OOB_score_list)

#save OOB mean and sd

OOB_mean_sd_data =  {'mean_OBB': mean_OOB, 'sd_OBB_sd' : sd_OOB}

OOB_mean_sd = pd.DataFrame(OOB_mean_sd_data, index=[0])

# OOB_mean_sd.to_csv(f'WE_Mean_OOB_{model}.csv', index = False)

print(f'\n\tMean OOB: {mean_OOB:.2f}\n')

# Metrics

y_test_we = Enc_Class_results['KnownSpecies']

y_pred_we = Enc_Class_results['Species_classification']

accuracy =  accuracy_score(y_test_we, y_pred_we)*100 
print(f"\tEvent Accuracy: {accuracy:.2f}%\n")

# Encounter Classification Report

classes = Enc_Class_results['KnownSpecies'].unique()

class_report_ev = classification_report(y_test_we, y_pred_we, output_dict=True, labels = classes)

class_report_ev = pd.DataFrame(class_report_ev).transpose()
class_report_ev['Labels'] = class_report_ev.index



#Matriz de confusão
cm = confusion_matrix(y_test_we, y_pred_we, labels = class_labels)

# Calculando a especificidade por classe
specificities = []
for i, label in enumerate(classes):
    # VN = Soma de todos os valores, exceto os da linha e coluna da classe atual
    tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
    # FP = Soma da coluna atual, exceto o valor da diagonal principal
    fp = cm[:, i].sum() - cm[i, i]
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    specificities.append(specificity)

# Calculate Macro and Weighted Averages for Specificity
macro_avg_specificity = np.mean(specificities)
weighted_avg_specificity = np.average(specificities, weights=class_report_ev.loc[classes, 'support'])


# Add specificities to DataFrame
specificities.extend([np.nan, macro_avg_specificity, weighted_avg_specificity])  # Add NaN for accuracy row

# Adicionando a especificidade ao DataFrame
class_report_ev['specificity'] = specificities

balanced_acc = balanced_accuracy_score(y_test_we, y_pred_we)
print(f"\tBalanced Accuracy: {balanced_acc:.2f}\n\t")


# Reorder columns to have labels first
class_report_ev_df = class_report_ev[['Labels', 'precision', 'recall', 'f1-score', 'specificity', 'support']]

# Criar uma nova linha com 'Labels' sendo 'balanced_accuracy' e 'precision' sendo o valor da balanced_accuracy
new_row = pd.DataFrame([['balanced_accuracy', balanced_acc] + [None] * (class_report_ev_df.shape[1] - 2)],
                       columns=class_report_ev_df.columns)

# Concatenar a nova linha no DataFrame existente
class_report_ev_df = pd.concat([class_report_ev_df, new_row], ignore_index=True)

print(class_report_ev)

# class_report_ev_df.to_csv(f'Class_report_WE_{model}_bal_acc.csv', index = False)





#%% CONFUSION MATRIX  - ENCOUNTER CLASSIFICATION 

# os.chdir(r'C:\Users\alebi\Desktop\1- RESULTADOS TESE 031124\7sp\LOGO\LOGO_B\CLASS_WE')#
os.chdir(r'C:\Users\alebi\Desktop\Tese Desktop LG\1- RESULTADOS TESE 031124 ASSOVIOS\7sp\CM_for_paper')
# Define classes
class_labels = ["D_delphis", "S_attenuata", "S_bredanensis", "S_clymene", "S_frontalis", "S_longirostris", "T_truncatus"]#,  "Globicephala"] # 'P_electra']
           #"P_crassidens",'F_attenuata'
           
conf_matrix = confusion_matrix(y_test_we, y_pred_we, labels = class_labels)

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
                       annot=  True,                                   # [[f"{val}%" for val in row] for row in cm_percent_sorted], 
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
# Save the figure
plt.show()
# plt.savefig(f'Conf_Matrix_WE_{model}_new.png', 
#             format='png', bbox_inches="tight", dpi=300)



#%% CLASS PROBS FOR WHISTLES 

os.listdir()

#Get the model n_estimators
model_n_estimators = Classifier_whistle.results['S_bredanensis']['Model_parameters']['n_estimators']


# Initialize an empty list to store DataFrames
Class_probs_LOGO = []

# Loop through all outer keys
for species, inner_dict in Classifier_whistle.results.items():
    # Initialize lists to store data
    indexes_flat = []
    class_probs_list = []
    group_id_flat = []
    
    # Loop through the inner lists (assuming they match in length)
    for idx_list, group_list, probs_df in zip(inner_dict['Indexes_raw'], inner_dict['EncounterNumber'], inner_dict['Class_probs']):
        # Append each index to the flat list
        indexes_flat.extend(idx_list)
        
        # Repeat the group_id for the length of the current index list
        group_id_flat.extend([group_list] * len(idx_list))
        
        # Append the class probabilities DataFrame to the list
        class_probs_list.append(probs_df)
    
    # Concatenate the class_probs DataFrames to ensure they're flattened into a single DataFrame
    class_probs_flat = pd.concat(class_probs_list, ignore_index=True)
    
    # Check if lengths match before creating DataFrame
    if len(indexes_flat) == len(group_id_flat) == len(class_probs_flat):
        # Create a DataFrame for the current outer_key
        df = pd.DataFrame({
            'UID': indexes_flat,
            'EncounterNumber': group_id_flat
        })
        
        # Combine the class_probs columns with the current DataFrame
        df = pd.concat([class_probs_flat, df], axis=1)# Place class_probs_df first
        
        # Add a column for the outer key
        df['KnownSpecies'] = species
        
        # Append the DataFrame to the list
        Class_probs_LOGO.append(df)
    else:
        print(f"Length mismatch for '{species}': "
              f"{len(indexes_flat)} (indexes), "
              f"{len(group_id_flat)} (group_id), "
              f"{len(class_probs_flat)} (class_probs)")

# Concatenate all DataFrames into a single DataFrame
if Class_probs_LOGO:  # Check if there are DataFrames to concatenate
    Class_probs_LOGO = pd.concat(Class_probs_LOGO, ignore_index=True)
    #print(final_df)
else:
    print("No DataFrames to concatenate.")
    


Class_probs_LOGO_for_evt = Class_probs_LOGO.copy()

# #Change col names 
Class_probs_LOGO.columns = [
    
        col + '_LOGO_WH' if i < 7 else col ##%%%%%%%%%%%%%%% CHANGE accordig to n labels 
        for i, col in enumerate(Class_probs_LOGO.columns)
                               ]
 
Class_probs_LOGO.info() 


os.chdir(r'D:\WHISTLE_CLASSIFIER_FINAL\7sp\LOGO\LOGO_PROBS_AVG_WH') 
  
Class_probs_LOGO.to_csv(f'CLASS_PROBS_WH_{model}.csv', index = False)
###############################################################################


#%% CLASS PROBS FOR WHISTLES ENCOUNTERS 

#Using this dataframe we will create the Class_Probs for Event Classifcation (for the Meta-Classifier) 

#Group the dataframe by encounter
Class_probs_LOGO_encs = Class_probs_LOGO_for_evt.groupby('EncounterNumber')


#loop through the encounters and create a list of dfs for results 
## Create a empty list for the dataframes of evt class probs

LOGO_Evt_Class_probs_list = []
for encounter, encounter_data in Class_probs_LOGO_encs:
    
    encounter_dataframe = pd.DataFrame(encounter_data)
    N_whistles = encounter_dataframe.shape[0]
    #First lets convert probs to absolute votes for enc total tree votes for each class ex: (columns 1 to 8) columns 0 to 8
    encounter_dataframe.iloc[:, 0:7] = encounter_dataframe.iloc[:, 0:7] #%%%%%%%%%%%%%%% CHANGE according to n labels
    
    #Get the sum tree for each class of the encounter 
    sum_tree = encounter_dataframe.iloc[:, 0: 7].sum() #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CHANGE according to n labels
    
    #calculate the encounter proportion of tree votes
    # total from the row sum_tree
    total_n_tree = sum(sum_tree)
    enc_tree_prop = sum_tree/total_n_tree
    #add the proportion tree to the dataframe
    
    #encounter_dataframe =  pd.concat([encounter_dataframe, enc_tree_prop.to_frame().T], ignore_index=True)
    
    #df with a repetead rows of the event class prob 
    
    encounter_class_prob = pd.DataFrame([enc_tree_prop] * encounter_dataframe.shape[0], 
                                        columns= encounter_dataframe.columns[:7]) ##%%%%%%%%%%%%%%% CHANGE accordig to n labels 
    
    encounter_class_prob.reset_index(drop=True, inplace=True)
        # Reset the index of both DataFrames

    encounter_dataframe.reset_index(drop=True, inplace=True)
    
    #concatenate the remain ed coluns (UID, KnownSpecies, EncounterNumber)
    enc_class_prob_filled = pd.concat([encounter_class_prob , encounter_dataframe.iloc[:, 7:]], axis = 1 ) ##%%%% CHANGE accordig to n labels 
    # Ensure you reset the index if necessary

    LOGO_Evt_Class_probs_list.append(enc_class_prob_filled)
    
Class_probs_Evts_LOGO = pd.concat(LOGO_Evt_Class_probs_list, ignore_index=True)    

# #Change col names 
Class_probs_Evts_LOGO.columns = [
    
        col + '_LOGO_WE' if i < 7 else col ##%%%%%%%%%%%%%%% CHANGE accordig to n labels 
        for i, col in enumerate(Class_probs_Evts_LOGO.columns)
                               ]

Class_probs_Evts_LOGO.info()

os.chdir(r'D:\WHISTLE_CLASSIFIER_FINAL\7sp\LOGO\LOGO_PROBS_AVG_WE')   
  
Class_probs_Evts_LOGO.to_csv(f'CLASS_PROBS_WE_{model}.csv', index = False)




'''
#%% GET A BALANCE TRAIN DATA FOR GRID SEARCH AND FIT THE LOGO CLASSIFIER FOR NEW INSTANCES 
os.chdir(File_path) 

#%% Function subsampling  

# Read Master_Whistle and define argments for the function

# os.chdir('C:\WHISTLE_CLASSIFIER_FINAL\Master_Whistle_7sp')##################### UD
# os.listdir()

#Master_Whistle = pd.read_csv('Master_Whistle_7sp.csv')##################### UD
Current_MW.info()
Current_MW['KnownSpecies'].unique()

# define the mastersheet
mastersheet = Current_MW
 
# define the factors list
factors_list = ['KnownSpecies','EncounterNumber']

# define the nbal  
minority_count = mastersheet.groupby('KnownSpecies').apply(lambda x : len(x), include_groups = False).min()   
nbal =  minority_count        

# define the species list (factor 1) 
factor1_list = mastersheet['KnownSpecies'].unique()


#%% Function subsampling that is called by the function training

def subsampling(df2subsample=pd.DataFrame(), nbal=1, factor_val = '', random_val = RS_bal): # ******* RS - same random_val from the classifier above
        if df2subsample.empty: df2subsample = mastersheet
        factors = factors_list
        factor1 = factors[0]
        factor2 = factors[1]

        if type(factor_val) == str:
            if factor_val == '': factor_val = 0
            else:                value = factor_val
        if type(factor_val) in [float,int]: 
            #factor1_list = balancing['factor1_list'] 
            #factor1_list = mastersheet[factor1].drop_duplicates().reset_index(drop=True)
            value = factor1_list[factor_val] #>>>species (value)

        #print('\tsubsampling...\t%s [r = %s]' %(value, random_val))
        filtered_ms = df2subsample[df2subsample[factor1] == value] #>>>filter for the species (factor1  = value)
        weights = pd.Series(filtered_ms.groupby(factor2)[factor1].transform('count')) # count the number of samples in each encounter of the species 
        weights1 = 1/weights
        #weights2 = 1/(weights/(weights.max()/2))
        weights2 = weights1/weights1.sum()
        
        subsample = filtered_ms.sample(n= nbal, weights = weights2, random_state= random_val, ignore_index = False)
        #.groupby([factor1,factor2])
        
       
        #print(subsample.apply(lambda x:len(x)))
        #len(n_subsample)
        return subsample
        # np.random.seed(seed)


#%% Function training

def training(data2train, print_summary = False):
        #nbal, factor1_list = self.find_minimum(df=data2train)
        print ('\t\t\tNBAL = %s'%(nbal))
        # loop to create the balanced train data dataframe 
        df_trained = pd.DataFrame()
        for ft in factor1_list:
            subsample = subsampling(df2subsample = data2train, nbal = nbal, factor_val = ft)
            df_trained  = pd.concat([df_trained,subsample],axis=0)
          
        
       #if print_summary: self.summarize_data(df=df_trained)
        #df_train = self.clear_columns(df=df_train)
        #return df_trained
        df_trained.to_csv(f'ALL_DATA_bal_{model}.csv', index = False)


#%% RUN FUNCTION FOR BALACING WEIGTH ALL DATA 
 

training(data2train = mastersheet)   

LOGO_bal = pd.read_csv(f'ALL_DATA_bal_{model}.csv')

LOGO_bal.info()

#%% CHECK ALL DATA BALANCED SUMMARY 

n_events_sp = LOGO_bal.groupby("KnownSpecies")['EncounterNumber'].nunique()
n_whistle_sp = LOGO_bal.groupby("KnownSpecies")['EncounterNumber'].count()

n_events_whistle_sp = pd.DataFrame({'n_events': n_events_sp, 'n_whistle': n_whistle_sp})

n_events_total = n_events_whistle_sp['n_events'].sum()

prop_events_sp = round(n_events_sp / n_events_total,2)


prop_whistle_sp = round ((n_whistle_sp / len(LOGO_bal)) *100, 2)

n_events_whistle_sp = pd.DataFrame({'n_events': n_events_sp, '% events': prop_events_sp, 'n_whistle': n_whistle_sp, 
                                   '% whistle': prop_whistle_sp})

print(n_events_whistle_sp)

n_species = len(LOGO_bal["KnownSpecies"].unique())
print(f'\n\n\tTotal number of species: {n_species }\n')

print(f'\n\tTotal number of events: {n_events_total}\n')

n_w= len(LOGO_bal)
print(f'\n\tTotal number of whistles: {n_w}\n\n')



#%


#%% END #######################################################################

#%% GRID SEARCH LOGO 

#READ THE TRAIN BAL DATA 

#Check the balanced data all encounters 


LOGO_bal.info() 

LOGO_bal.isnull().values.any()



#%% CREATE TRAIN DATA X (X_train and y_train) 

##select true class and features of the train data 
X_train = LOGO_bal.loc[:, 'FREQMAX':'STEPDUR']
X_train.info()
X_train.shape

#select target from the train data 
y_train = LOGO_bal.loc[:, 'KnownSpecies']
print(f'y_train shape: {y_train.shape}')


#%% DEFINE RF MODEL, CV METHOD AND PARAM GRID 

#Define Rf model 
rf_classifier = RandomForestClassifier(random_state= RS_rf, n_jobs= -1)

# Define cross-validation method
logo = LeaveOneGroupOut()

#Define grid for search 
param_grid_final = {
    'n_estimators': [100, 500, 1000],
    'max_depth': [None, 10, 20, 30],
    'max_features': ['sqrt', 'log2', 10, 0.5],
}

#%% DEFINE THE GROUPS AND GRID SEARCH CV TO RUN 

groups = LOGO_bal['EncounterNumber']

#change according to the param_grid to be used
grid_search_LOGO = GridSearchCV(estimator = rf_classifier, param_grid = param_grid_final, 
                                       cv=  logo, 
                                       scoring=['accuracy', 'f1_macro', 'f1_weighted'],
                                       refit='f1_weighted',  # Will refit the model with the best F1-score
                                       n_jobs=-1, verbose= 1,
                                       return_train_score=False)# default - DO NOT Include training scores in the results

#%% START PROCESSING GRID SEARCH

start_time = time.time()

######Fit the grid search to the data##########
grid_search_LOGO.fit(X_train, y_train, groups = groups)#change according to the param_grid to be used
########################################

end_time = time.time()

elapsed_time = (end_time - start_time)/60
print(f"Elapsed time: {elapsed_time} minutes") 

#%% SAVE RESULTS 


# Get the results of the grid search
results = grid_search_LOGO.cv_results_

# Convert cv_results_ into a DataFrame
results_df = pd.DataFrame(results)

results_df.columns 

#Filter columns with results 
selected_columns = ['param_max_depth', 'param_max_features', 'param_n_estimators',
                    'mean_test_accuracy', 'std_test_accuracy', 'rank_test_accuracy',
                    'mean_test_f1_macro', 'std_test_f1_macro', 'rank_test_f1_macro',
                    'mean_test_f1_weighted', 'std_test_f1_weighted', 'rank_test_f1_weighted']

results_grid = results_df[selected_columns]

# Save the DataFrame to a CSV file
results_grid.to_csv(f'Grid_Search_{model}.csv', index=False)

print("Grid search results saved")


##################################################
#%%  
'''