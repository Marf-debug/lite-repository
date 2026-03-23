import pandas as pd
import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt

from sqlalchemy import create_engine
from urllib.parse import quote_plus as urlquote
from ipywidgets import  GridspecLayout, Layout
from matplotlib_venn import venn2, venn2_circles, venn2_unweighted,venn3, venn3_circles
import qgrid

# connection to DB
def return_cohort(username,password,cohort_type=0):
    # Postgres username, password, and database name
    POSTGRES_ADDRESS = 'alhs-data.validitron.io' ## INSERT YOUR DB ADDRESS IF IT'S NOT ON PANOPLY
    POSTGRES_PORT = '5432'
    POSTGRES_DBNAME = 'lhscoursedb' ## CHANGE THIS TO YOUR DATABASE NAME

    POSTGRES_USERNAME = username ## CHANGE THIS TO YOUR PANOPLY/POSTGRES USERNAME
    POSTGRES_PASSWORD = urlquote(password)#urlquote ## CHANGE THIS TO YOUR PANOPLY/POSTGRES PASSWORD 

    postgres_str = ('postgresql://{username}:{password}@{ipaddress}:{port}/{dbname}'.format(username=POSTGRES_USERNAME,password=POSTGRES_PASSWORD,ipaddress=POSTGRES_ADDRESS,port=POSTGRES_PORT,dbname=POSTGRES_DBNAME))
    #Create the connection
    cnx = create_engine(postgres_str)
    # there are 2 types of cohort in the database: 0 - data without any transformation, 1 - data without extrime values (+- 3*iqr)
    if cohort_type==0:
        dataset=pd.read_sql_query('''SELECT * FROM biogrid_vaed.RMH_COHORT;''', cnx)
    elif cohort_type==1:
        dataset=pd.read_sql_query('''SELECT * FROM biogrid_vaed.RMH_COHORT_WITHOUT_OUTLIERS;''', cnx)
    elif cohort_type==2:
        dataset=pd.read_sql_query('''SELECT * FROM biogrid_vaed.WHOLE_COHORT_WITHOUT_OUTLIERS;''', cnx)
    # Coerce visit/admission counters to numeric if present (supports both 2018/2017 and 2025/2024 schemas)
    for col in ['visits_b2018', 'admissions_2017', 'visits_b2025', 'admissions_2024']:
        if col in dataset.columns:
            dataset[col] = pd.to_numeric(dataset[col], errors='coerce')
    return dataset
    
# qgrid customisation 
def dataframe_2_qgrid(df):

    col_opts = {'editable': False}

    grid_options={'forceFitColumns': False, 
              'defaultColumnWidth': 220,'highlightSelectedCell': True }
              
    column_definitions={ 'index': { 'maxWidth': 0, 'minWidth': 0, 'width': 0 }, 
                         'pkey':  { 'maxWidth': 0, 'minWidth': 0, 'width': 0 }, 
                         'patient_id':  { 'maxWidth': 0, 'minWidth': 0, 'width': 0 }, 
                         #'pkey':  { 'toolTip': "Patient's identifier"},
                         'age':  { 'toolTip': "Age as of 01/01/2018, all petients age 85+ were joined into one category '86'"},
                         'sex':  { 'toolTip': "Column gender has 2 categories: M, F"},
                         'interpreter    ':  { 'toolTip': "Whether the patient requires an interpreter"},
                         'interpreter':  { 'toolTip': "Whether the patient requires an interpreter"},
                         'hypertension':  { 'toolTip': "Whether person has a history of treatment for hypertension"},
                         'hyperlidipeamia':  { 'toolTip': "Whether person has a history of treatment for hyperlipidaemia - Yes/No"},
                         'hyperlipidaemia':  { 'toolTip': "Whether person has a history of treatment for hyperlipidaemia - Yes/No"},
                         'ischemic_heart_disease':  { 'toolTip': "Whether person has a history of treatment for ischemic heart disease - Yes/No"},
                         'cardiac_failure':  { 'toolTip': "Whether person has a history of treatment for cardiac failure - Yes/No"},
                         'neuropathy':  { 'toolTip': "Whether person has a history of treatment for neuropathy - Yes/No"},
                         'nephropathy':  { 'toolTip': "Whether person has a history of treatment for nephropathy - Yes/No"},
                         'oral_contraceptive':  { 'toolTip': "On Oral Contraceptive - Yes/No"},
                         'beta_blockers':  { 'toolTip': "On Beta Blockers - Yes/No"},
                         'ace_inhibitor':  { 'toolTip': "On ACE Inhibitor - Yes/No"},
                         'calcium_chan_block':  { 'toolTip': "On Calcium Channel Blockers - Yes/No"},
                         'corticosteroids':  { 'toolTip': "On Corticosteroids - Yes/No"},
                         'thaizide': { 'toolTip': "On Thiazide - Yes/No"},
                         'agr2receptorblock': { 'toolTip': "On angiotensin 2 Receptor Blockers - Yes/No"},
                         'aspirin': { 'toolTip': "On aspirin treatment - Yes/No"},
                         'familyhistory': { 'toolTip': "Family history of diabetes"},
                         'siblings': { 'toolTip': "Patient has siblings with diabeties"},
                         'children': { 'toolTip': "Patient has children with diabeties"},
                         'gestdiabetes': { 'toolTip': "Whether the person has previously been diagnosed with gestational diabetes"},
                         'symptomaticpn': { 'toolTip': "Whether the patient has a history of symptomatic peripheral neuropathy"},
                         'depression': { 'toolTip': "Whether the person has a history of depression"},
                         'isletantibody': { 'toolTip': "Whether the person has had screening for Islet Antibody"},
                         'coeliacantibody': { 'toolTip': "Whether the person has had screening for Coeliac Antibody"},
                         'tfunction': { 'toolTip': "Whether the person has had screening for Thyroid function"},
                         'tantibody': { 'toolTip': "Whether the person has had screening for Thyroid Antibody"},
                         'vitaminb12': { 'toolTip': "Whether the person has had screening for Vitamin B12"},
                         'dka_diagnosis': { 'toolTip': "Whether the person has diagnosis of diabetic ketoacidosis "},
                         'alcohol': { 'toolTip': "Whether the person drinks alcohol"},
                         'smoker': { 'toolTip': "Smoking status of the person. The column has 4 categories: CURRENT, NO, UNKNOWN, FORMER"},
                         'firstdiagnosed': { 'toolTip': "Number of years since the person has been diagnosed with diabetes type 2 as of 01/01/2018"},
                         'sodium': { 'toolTip': "The person's measured sodium level in blood (mmol/L)."},
                         'potassium': { 'toolTip': "The person's measured potassium level in blood (mmol/L)."},
                         'chloride': { 'toolTip': "The person's measured chloride level in blood (mmol/L)."},
                         'bicarbonate': { 'toolTip': "The person's measured bicarbonate level in blood (mmol/L)."},
                         'creatinine': { 'toolTip': "The person's measured serum creatinine level in blood (mmol/L)."},
                         'urea': { 'toolTip': "The person's measured urea level in blood (mmol/L)."},
                         'haemoglobin': { 'toolTip': "The person's measured haemoglobin level (g/L)."},
                         'wcc': { 'toolTip': "The person's measured white cell count (x10^9/L)."},
                         'platelets': { 'toolTip': "The person's measured platelets count (x10^9/L)."},
                         'rcc': { 'toolTip': "The person's measured red blood cell count (x10^12/L)."},
                         'pcv': { 'toolTip': "Person's measured Packed Cell Volume."},
                         'mcv': { 'toolTip': "Person's measured Mean Cell volume (fl)."},
                         'hba1c': { 'toolTip': "The person's measured glycosylated haemoglobin (HbA1c) level as a percentage."},
                         'alb_crratio': { 'toolTip': "The patients albumin creatinine ratio in urine (mg/moll)"},
                         'egfr': { 'toolTip': "The patients albumin creatinine ratio in urine (mg/moll)"},
                         'weight': { 'toolTip': "Weight at booking (kg)."},
                         'weight_more_100': { 'toolTip': "Indicates whether the person's weight is higher than 100kg."},
                         'waist': { 'toolTip': "Waist circumference (cm)"},
                         'systolic': { 'toolTip': "The person's systolic blood pressure"},
                         'diastolic': { 'toolTip': "The person's diastolic blood pressure"},
                         'elevated_bp': { 'toolTip': "Indicates whether the person has elevated blood pressure ( > 190/95 mmHg)"},
                         'bmi': { 'toolTip': "The patient's body mass index."},
                         'obesity': { 'toolTip': "Flag for whether the patient is obese"},
                         'retinopathy': { 'toolTip': "Whether the person has retinopathy (either eye)"},
                         'charcot': { 'toolTip': "Whether the person has Charcot foot (either foot)"},
                         'ulceration': { 'toolTip': "Status of whether the person has a current foot ulcer on either foot"},
                         'stroke': { 'toolTip': "Whether the person had cerebral stroke due to vascular disease."},
                         'ami': { 'toolTip': "Whether the person had an acute myocardial infarction."},
                         'tia': { 'toolTip': "Whether the person has had transient ischemic attack."},
                         'claudication': { 'toolTip': "Whether the person experienced claudication."},
                         'cabg': { 'toolTip': "Whether the person has had a coronary artery bypass graft"},
                         'cardiomyopathy': { 'toolTip': "Whether the person has had ischaemic cardiomyopathy/cardiac failure."},
                         'neuropathy_ep': { 'toolTip': "Whether the person has had autonomic neuropathy"},
                         'num_of_surv_in_2017': { 'toolTip': "Number of appointments before 01/01/2018"},
                         'method_manage_dt2': { 'toolTip': "The method being used to manage the person's diabetes. The column has 7 categories: BOTH INSULIN AND TABS, INSULIN,  INSULIN AND NON-INSULIN, OTHER, TABLETS, NON-INSULIN, DIET ONLY"},
                         'ind_of_nephropathy': { 'toolTip': "Whether the person has nephropathy (based on: eGFR, UACR indicators)"},
                         'cardiovascular_disease': { 'toolTip': "Whether the person has cardiovascular disease (based on: MI, ischaemic cardiomyopathy/cardiac failure, CABG)"},
                         'cerebrovascular_disease': { 'toolTip': "Whether the person has cerebrovascular disease (based on: AIT, stroke indicators)"},
                         'lower_limb_problems':  { 'toolTip': "Whether the person has lower limb problems like Charcot foot, ulceration "},
                         'number_of_admissions_in_2017': { 'toolTip': "Number of admissions to the hospital during 2017"},
                         'outcome': { 'toolTip': "Whether the person was hospitalised after 01/01/2018"},


                         'visits_b2018': { 'toolTip': "Number of appointments before the baseline period (2018 schema)"},
                         'admissions_2017': { 'toolTip': "Number of hospital admissions in the baseline year (2017 schema)"},
                         'visits_b2025': { 'toolTip': "Number of appointments before the baseline period (2025 schema)"},
                         'admissions_2024': { 'toolTip': "Number of hospital admissions in the baseline year (2024 schema)"},
                         'ed_presentations_next_year': { 'toolTip': "Number of ED presentations in the next year"},
                         'any_ed_next_year': { 'toolTip': "Whether the person had any ED presentation in the next year (Yes/No)"},
                         'comorb_count': { 'toolTip': "Count of comorbidities (derived feature)"},
                         'comorb_ge4': { 'toolTip': "Whether comorbidity count is >= 4 (Yes/No)"},
                         'polypharm_cv_n': { 'toolTip': "Number of cardiovascular-related medicines (derived feature)"},
                         'polypharm_cv_ge3': { 'toolTip': "Whether cardiovascular polypharmacy is >= 3 (Yes/No)"},
                         'on_insulin': { 'toolTip': "Whether the person is on insulin (Yes/No)"},
                         'insulin_plus3cv': { 'toolTip': "Whether on insulin and cardiovascular polypharmacy >= 3 (Yes/No)"},
                         'hba1c_gt9': { 'toolTip': "Whether HbA1c is > 9% (Yes/No)"},
                         'egfr_lt45': { 'toolTip': "Whether eGFR is < 45 (Yes/No)"},
                         'highrisk_glyco_renal': { 'toolTip': "Glyco-renal high risk flag (derived feature)"},
                         'glyco_renal_highrisk': { 'toolTip': "Glyco-renal high risk flag (derived feature)"},
                         'acr': { 'toolTip': "Albumin-to-creatinine ratio (ACR)"},
                         'risk_hba1c': { 'toolTip': "Risk component: HbA1c"},
                         'risk_egfr': { 'toolTip': "Risk component: eGFR"},
                         'risk_acr': { 'toolTip': "Risk component: ACR"},
                         'risk_bp': { 'toolTip': "Risk component: blood pressure"},
                         'risk_potassium': { 'toolTip': "Risk component: potassium"},
                         'risk_prior_adm': { 'toolTip': "Risk component: prior admissions"},
                         'risk_foot_ulcer': { 'toolTip': "Risk component: foot ulcer"},
                         'risk_cardiac': { 'toolTip': "Risk component: cardiac history"},
                         'risk_duration': { 'toolTip': "Risk component: diabetes duration"},
                         'risk_age': { 'toolTip': "Risk component: age"},
                         'risk_bmi': { 'toolTip': "Risk component: BMI"},
                         'risk_score_2025': { 'toolTip': "Composite risk score (2025)"},
                         'prob_hosp_2025': { 'toolTip': "Predicted probability of hospitalisation (2025)"},
                         'label': { 'toolTip': "Outcome label (target)"},
                         'diabetes_group': { 'toolTip': "Diabetes subgroup / cluster label (derived feature)"},
                         'risk_ulcer': { 'toolTip': "Derived ulcer risk indicator"},
                         'risk_instability': { 'toolTip': "Derived clinical instability risk indicator"},

                               }
          
    gri=qgrid.show_grid(df,column_options=col_opts, grid_options=grid_options, column_definitions=column_definitions)
    return gri
    
# Function to Detection Outlier on one-dimentional datasets.
# For nurmally distributed data (Z-score treatment)
def find_anomalies(data):
    #define a list to accumlate anomalies
    anomalies = []
    
    # Set upper and lower limit to 3 standard deviation
    random_data_std = np.std(data)
    random_data_mean = np.mean(data)
    anomaly_cut_off = random_data_std * 3
    
    lower_limit  = random_data_mean - anomaly_cut_off 
    upper_limit = random_data_mean + anomaly_cut_off
    #print(lower_limit)
    # Generate outliers
    for outlier in data:
        if outlier > upper_limit or outlier < lower_limit:
            anomalies.append(outlier)
    return anomalies, upper_limit, lower_limit

# Function to Detection Outlier on one-dimentional datasets.
# IQR based filtering  
def find_outliers(df, col_name):   
    percentile25 = df[col_name].quantile(0.25)
    percentile75 = df[col_name].quantile(0.75)
    
    iqr = percentile75-percentile25
    upper_limit = percentile75 + 1.5 * iqr
    lower_limit = percentile25 - 1.5 * iqr
    
    anomalies_upper = df[col_name][df[col_name] > upper_limit]
    anomalies_lower = df[col_name][df[col_name] < lower_limit]
    anomalies = list(pd.concat([anomalies_upper, anomalies_lower]))
    return anomalies, upper_limit, lower_limit
    
#Dictionary for dataset desription
col_name_2_discription = {'pkey': "Patient's identifier",
 'patient_id': "Patient identifier (new schema)",
 'age': "Age as of 01/01/2018, all petients age 85+ were joined into one category '86'",
 'sex': "Column gender has 2 categories: M, F",
 'interpreter': "Whether the patient requires an interpreter",
 'hypertension': "Whether person has a history of treatment for hypertension",
 'hyperlidipeamia': "Whether person has a history of treatment for hyperlipidaemia - Yes/No",
 'hyperlipidaemia': "Whether person has a history of treatment for hyperlipidaemia - Yes/No",
 'ischemic_heart_disease': "Whether person has a history of treatment for ischemic heart disease - Yes/No",
 'cardiac_failure': "Whether person has a history of treatment for cardiac failure - Yes/No",
 'neuropathy': "Whether person has a history of treatment for neuropathy - Yes/No",
 'nephropathy': "Whether person has a history of treatment for nephropathy - Yes/No",
 'oral_contraceptive': "On Oral Contraceptive - Yes/No",
 'beta_blockers': "On Beta Blockers - Yes/No",
 'ace_inhibitor': "On ACE Inhibitor - Yes/No",
 'calcium_chan_block': "On Calcium Channel Blockers - Yes/No",
 'corticosteroids': "On Corticosteroids - Yes/No",
 'thaizide': "On Thiazide - Yes/No",
 'agr2receptorblock': "On angiotensin 2 Receptor Blockers - Yes/No",
 'aspirin': "On aspirin treatment - Yes/No",
 'familyhistory': "Family history of diabetes",
 'siblings': "Patient has siblings with diabeties",
 'children': "Patient has children with diabeties",
 'gestdiabetes': "Whether the person has previously been diagnosed with gestational diabetes",
 'symptomaticpn': "Whether the patient has a history of symptomatic peripheral neuropathy",
 'depression': "Whether the person has a history of depression",
 'isletantibody': "Whether the person has had screening for Islet Antibody",
 'coeliacantibody': "Whether the person has had screening for Coeliac Antibody",
 'tfunction': "Whether the person has had screening for Thyroid function",
 'tantibody': "Whether the person has had screening for Thyroid Antibody",
 'vitaminb12': "Whether the person has had screening for Vitamin B12",
 'dka_diagnosis': "Whether the person has diagnosis of diabetic ketoacidosis ",
 'alcohol': "Whether the person drinks alcohol",
 'smoker': "Smoking status of the person",
 'years_diagnosed': "Number of years since the person has been diagnosed with diabetes type 2 as of 01/01/2018",
 'sodium': "The person's measured sodium level in blood (mmol/L)",
 'potassium': "The person's measured potassium level in blood (mmol/L)",
 'chloride': "The person's measured chloride level in blood (mmol/L)",
 'bicarbonate': "The person's measured bicarbonate level in blood (mmol/L)",
 'creatinine': "The person's measured serum creatinine level in blood (mmol/L)",
 'urea': "The person's measured urea level in blood (mmol/L)",
 'haemoglobin': "The person's measured haemoglobin level (g/L)",
 'wcc': "The person's measured white cell count (x10^9/L)",
 'platelets': "The person's measured platelets count (x10^9/L)",
 'rcc': "The person's measured red blood cell count (x10^12/L)",
 'pcv': "Person's measured Packed Cell Volume",
 'mcv': "Person's measured Mean Cell volume (fl)",
 'hba1c': "The person's measured glycosylated haemoglobin (HbA1c) level as a percentage",
 'alb_crratio': "The patients albumin creatinine ratio in urine (mg/moll)",
 'egfr': "The patients albumin creatinine ratio in urine (mg/moll)",
 'weight': "Weight at booking (kg)",
 'weight_more_100': "Indicates whether the person's weight is higher than 100kg",
 'waist': "Waist circumference (cm)",
 'systolic': "The person's systolic blood pressure",
 'diastolic': "The person's diastolic blood pressure",
 'elevated_bp': "Indicates whether the person has elevated blood pressure ( > 190/95 mmHg)",
 'bmi': "The patient's body mass index",
 'obesity': "Flag for whether the patient is obese",
 'retinopathy': "Whether the person has retinopathy (either eye)",
 'charcot': "Whether the person has Charcot foot (either foot)",
 'ulceration': "Status of whether the person has a current foot ulcer on either foot",
 'stroke': "Whether the person had cerebral stroke due to vascular disease",
 'ami': "Whether the person had an acute myocardial infarction",
 'tia': "Whether the person has had transient ischemic attack",
 'claudication': "Whether the person experienced claudication",
 'cabg': "Whether the person has had a coronary artery bypass graft",
 'cardiomyopathy': "Whether the person has had ischaemic cardiomyopathy/cardiac failure",
 'neuropathy_ep': "Whether the person has had autonomic neuropathy",
 'visits_b2018': "Number of appointments before the baseline period (2018 schema)",
 'visits_b2025': "Number of appointments before the baseline period (2025 schema)",
 'method_manage_dt2': "The method is used to manage the person's diabetes",
 'ind_of_nephropathy': "Whether the person has nephropathy (based on: eGFR, UACR indicators)",
 'cardiovascular_disease': "Whether the person has cardiovascular disease (based on: MI, ischaemic cardiomyopathy/cardiac failure, CABG)",
 'cerebrovascular_disease': "Whether the person has cerebrovascular disease (based on: AIT, stroke indicators)",
 'lower_limb_problems': "Whether the person has lower limb problems like Charcot foot, ulceration ",
 'admissions_2017': "Number of admissions to the hospital during the baseline year (2017 schema)",
 'admissions_2024': "Number of admissions to the hospital during the baseline year (2024 schema)",
 'outcome': "Whether the person was hospitalised after 01/01/2018 - Yes/No",
 'hospital': "Hospital where a patient was admitted: RMH, WH",
 'albumin': "The persons measured albumin level (g/L)",

 'ed_presentations_next_year': "Number of ED presentations in the next year",
 'any_ed_next_year': "Whether the person had any ED presentation in the next year (Yes/No)",
 'comorb_count': "Count of comorbidities (derived feature)",
 'comorb_ge4': "Whether comorbidity count is >= 4 (Yes/No)",
 'polypharm_cv_n': "Number of cardiovascular-related medicines (derived feature)",
 'polypharm_cv_ge3': "Whether cardiovascular polypharmacy is >= 3 (Yes/No)",
 'on_insulin': "Whether the person is on insulin (Yes/No)",
 'insulin_plus3cv': "Whether on insulin and cardiovascular polypharmacy >= 3 (Yes/No)",
 'hba1c_gt9': "Whether HbA1c is > 9% (Yes/No)",
 'egfr_lt45': "Whether eGFR is < 45 (Yes/No)",
 'highrisk_glyco_renal': "Glyco-renal high risk flag (derived feature)",
 'glyco_renal_highrisk': "Glyco-renal high risk flag (derived feature)",
 'acr': "Albumin-to-creatinine ratio (ACR)",
 'risk_hba1c': "Risk component: HbA1c",
 'risk_egfr': "Risk component: eGFR",
 'risk_acr': "Risk component: ACR",
 'risk_bp': "Risk component: blood pressure",
 'risk_potassium': "Risk component: potassium",
 'risk_prior_adm': "Risk component: prior admissions",
 'risk_foot_ulcer': "Risk component: foot ulcer",
 'risk_cardiac': "Risk component: cardiac history",
 'risk_duration': "Risk component: diabetes duration",
 'risk_age': "Risk component: age",
 'risk_bmi': "Risk component: BMI",
 'risk_score_2025': "Composite risk score (2025)",
 'prob_hosp_2025': "Predicted probability of hospitalisation (2025)",
 'label': "Outcome label (target)",
 'diabetes_group': "Diabetes subgroup / cluster label (derived feature)",
 'risk_ulcer': "Derived ulcer risk indicator",
 'risk_instability': "Derived clinical instability risk indicator"
}

#Dicrionary for data types 
pyttype_2_vartype={'object': 'Categorical', 'float64': 'Numerical', 'int64': 'Boolean'}

def features_description(dataframe, category):
    feature_name=[]
    feature_type=[]
    feature_discr=[]
    for feature in dataframe.columns:
        feature_name.append(feature)
        feature_type.append(pyttype_2_vartype[str(dataframe[feature].dtypes)])
        feature_discr.append(col_name_2_discription.get(feature, 'No description available'))
    d = {'column name': feature_name, 'type': feature_type , 'description': feature_discr}
    df = pd.DataFrame(data=d)    
    if category!='All':
        return df[(df['type']==category)].sort_values(by=['type', 'column name'],ignore_index=True)
    else:
        return df.sort_values(by=['type', 'column name'],ignore_index=True)
        
    
def label_2_name(data, feature_name):
    new_name=feature_name+'_str'
    data.loc[(data[feature_name]==1),new_name ] = 'Admitted'
    data.loc[(data[feature_name]==0),new_name ] = 'Non-admitted'
    return data