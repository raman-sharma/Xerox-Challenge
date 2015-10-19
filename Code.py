import pandas as pd
import numpy as np
import math

id_age_train = pd.read_csv("E:\\ml comp\\xerox machine learning challenge\\training_data\\id_age_train.csv")
id_label_train = pd.read_csv("E:\\ml comp\\xerox machine learning challenge\\training_data\\id_label_train.csv")
labs_train = pd.read_csv("E:\\ml comp\\xerox machine learning challenge\\training_data\\id_time_labs_train.csv")
vitals_train = pd.read_csv("E:\\ml comp\\xerox machine learning challenge\\training_data\\id_time_vitals_train.csv")
#labs_train = id_time_labs_train.drop(["ID","TIME"],axis=1)
#Merging the three data frames
#features = pd.concat([vitals_train,labs_train],axis=1)
#features = features.merge(id_age_train)
#features = features.merge(id_label_train)

id_age_valid = pd.read_csv("E:\\ml comp\\xerox machine learning challenge\\validation_data\\id_age_val.csv")
labs_valid = pd.read_csv("E:\\ml comp\\xerox machine learning challenge\\validation_data\\id_time_labs_val.csv")
vitals_valid = pd.read_csv("E:\\ml comp\\xerox machine learning challenge\\validation_data\\id_time_vitals_val.csv")

#Vitals Preprocessing V1-V5 are processed according to the 2-way moving averages and nan's are replaced with moving averages value
vitals_map_normal = {
                'V1':110,
                'V2':60,
                'V3':70,
                'V4':20,
                'V5':95,
                'V6':98.7
                }
############ To Do :- impute these values in copied dataframe instead of vitals_map_normal ############
vitals_map_mean = {
                'V1':vitals_train.V1.mean(),
                'V2':vitals_train.V2.mean(),
                'V3':vitals_train.V3.mean(),
                'V4':vitals_train.V4.mean(),
                'V5':vitals_train.V5.mean(),
                'V6':vitals_train.V6.mean()
                    }

def training_vitals_imputation(vitals_train):
    vitals = ['V1','V2','V3','V4','V5','V6']
    vitals_train_temp = vitals_train.copy()
    for vital in vitals:
        vitals_train_temp[vital] = vitals_train_temp[vital].fillna(vitals_map_normal[vital])

    vitals_train_temp = pd.rolling_mean(vitals_train_temp,2)
    
    for vital in vitals:
        null_index = vitals_train[vitals_train[vital].isnull()].index
        null_list = vitals_train_temp[vital][vitals_train[vitals_train[vital].isnull()].index].values
        vitals_train[vital].loc[null_index] = null_list
    # Handle exceptional case of 1st entry of V4 in vital_train dataframe
    vitals_train['V4'].fillna(20,inplace=True)
    vitals_train['V6'].fillna(98.7,inplace=True)
    return vitals_train
    
training_vitals = training_vitals_imputation(vitals_train)
training_vitals.save("training_vitals_values.pkl")

lab_map_normal = {
                'L1':7.40,   # Arterial Blood Ph
                'L2':42,     # Partial Pressure of CO2
                'L3':100,    # Partial Pressure of O2
                'L4':140,    # Sodium Content
                'L5':4.35,   # Potassium Content
                'L6':26,     # Bicarbonate Content
                'L7':15,     # Blood Urea Nitrogen
                'L8':0.8,     # Serum Creatinine
                'L9':170,     # Can't understand...So replacing by mean
                'L10':48,     # Hematocrit percentage
                'L11':275,    # Platelet Count 
                'L12':1.1,    # Bilirubin Content
                'L13':1600,   # Urine Output
                'L14':90,     # LDL Cholesterol
                'L15':2,      # Lactic Acid 
                'L16':5,      # Troponin I(mean)
                'L17':0.38,   # Troponin T(mean)
                'L18':125,    # Random Blood Glucose
                'L19':100,    # Fasting Blood Glucose
                'L20':21,     # FIO2
                'L21':4.4,    # Albumin
                'L22':100,    # Alkaline phosphatase
                'L23':25,     # alanine
                'L24':60,     # HDL Cholesterol
                'L25':2,      # Magnesium
}
labs_list = ['L1','L2','L3','L4','L5','L6','L7','L8','L9','L10',
             'L11','L12','L13','L14','L15','L16','L17','L18',
             'L19','L20','L21','L22','L23','L24','L25']
def training_lab_imputation():
    for idx in labs_train.ID.unique():
        #id_df = labs_train[labs_train.ID==idx]
        #temp_df = id_df.copy()
        for lab in labs_list:
            #print lab
            id_df_lab = labs_train[labs_train.ID==idx][lab]
            id_df_lab_temp = id_df_lab.copy()
            group_mean_lab = id_df_lab_temp.mean()
            if (math.isnan(group_mean_lab)):
                id_df_lab_temp.fillna(lab_map_normal[lab],inplace=True)
            else:
                id_df_lab_temp.fillna(group_mean_lab,inplace=True)
            # Now calculating the moving average in the series 
            id_df_lab_temp = pd.rolling_mean(id_df_lab_temp,2)
            # replace the nan values in id_df_lab with calculated rolled mean values in id_df_lab_temp
            null_index = id_df_lab[id_df_lab.isnull()].index
            null_list = id_df_lab_temp[id_df_lab[id_df_lab.isnull()].index].values
            id_df_lab.loc[null_index] = null_list
            #if first value in series is nan replaceit with default
            if (math.isnan(id_df_lab[id_df_lab.index[0]])):
                id_df_lab[id_df_lab.index[0]] = lab_map_normal[lab]
            #replace original labs_train with the series
            #labs_train[labs_train.ID==idx][lab] = id_df_lab
            labs_train.loc[labs_train[labs_train.ID==idx].index,lab]=  id_df_lab
    return labs_train
        
labs_train.save("training_lab_values.pkl")
