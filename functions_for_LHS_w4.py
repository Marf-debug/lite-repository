import pandas as pd
import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt

from sqlalchemy import create_engine
from urllib.parse import quote_plus as urlquote
from ipywidgets import  GridspecLayout, Layout
from matplotlib_venn import venn2, venn2_circles, venn2_unweighted,venn3, venn3_circles
import qgrid
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef,accuracy_score,make_scorer,balanced_accuracy_score,classification_report,roc_auc_score,confusion_matrix,RocCurveDisplay,f1_score,precision_score,recall_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB,BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from imblearn.pipeline import Pipeline
from itertools import combinations
from sklearn.base import clone
import matplotlib.gridspec as gridspec
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, auc, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
from numpy import interp
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import RocCurveDisplay
from IPython.display import display_html 

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
        
    dataset = dataset.astype({'visits_b2018': 'float64', 'admissions_2017': 'float64'})
    return dataset
    
# qgrid customisation 
def dataframe_2_qgrid(df):

    col_opts = {'editable': False}

    grid_options={'forceFitColumns': False, 
              'defaultColumnWidth': 220,'highlightSelectedCell': True }
              
    column_definitions={ 'index': { 'maxWidth': 0, 'minWidth': 0, 'width': 0 }, 
                         'pkey':  { 'maxWidth': 0, 'minWidth': 0, 'width': 0 }, 
                         #'pkey':  { 'toolTip': "Patient's identifier"},
                         'age':  { 'toolTip': "Age as of 01/01/2018, all petients age 85+ were joined into one category '86'"},
                         'sex':  { 'toolTip': "Column gender has 2 categories: M, F"},
                         'interpreter    ':  { 'toolTip': "Whether the patient requires an interpreter"},
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
 'visits_b2018': "Number of appointments before 01/01/2018",
 'method_manage_dt2': "The method is used to manage the person's diabetes",
 'ind_of_nephropathy': "Whether the person has nephropathy (based on: eGFR, UACR indicators)",
 'cardiovascular_disease': "Whether the person has cardiovascular disease (based on: MI, ischaemic cardiomyopathy/cardiac failure, CABG)",
 'cerebrovascular_disease': "Whether the person has cerebrovascular disease (based on: AIT, stroke indicators)",
 'lower_limb_problems': "Whether the person has lower limb problems like Charcot foot, ulceration ",
 'admissions_2017': "Number of admissions to the hospital during 2017",
 'outcome': "Whether the person was hospitalised after 01/01/2018 - Yes/No",
 'hospital': "Hospital where a patient was admitted: RMH, WH",
 'albumin': "The persons measured albumin level (g/L)"}

#Dicrionary for data types 
pyttype_2_vartype={'object': 'Categorical', 'float64': 'Numerical', 'int64': 'Boolean'}

def features_description(dataframe, category):
    feature_name=[]
    feature_type=[]
    feature_discr=[]
    for feature in dataframe.columns:
        feature_name.append(feature)
        feature_type.append(pyttype_2_vartype[str(dataframe[feature].dtypes)])
        feature_discr.append(col_name_2_discription[feature])
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
    
def cohort_2_transform_df(X_train, X_test, scaler = True):
    values_num = X_train.dtypes != object
    values_cat = X_train.dtypes == object

    si_0 = SimpleImputer(missing_values=np.nan,strategy='median') 
    ss = StandardScaler() 
    ohe = OneHotEncoder(handle_unknown = 'ignore') #for extrimly unbalanced cases
    # define column groups with same processing
    cat_vars = values_cat
    num_vars = values_num
    # set up pipelines for each column group
    categorical_pipe = Pipeline([('ohe', ohe)])
    
    if scaler:
        numeric_pipe = Pipeline([('si_0', si_0), ('ss', ss)])
    else:
        numeric_pipe = Pipeline([('si_0', si_0)])
    # set up columnTransformer
    col_transformer = ColumnTransformer(
                        transformers=[
                            ('nums', numeric_pipe, num_vars),
                            ('cats', categorical_pipe, cat_vars)
                        ],
                        remainder='drop',
                        n_jobs=-1
                        )


    X_train_np = col_transformer.fit_transform(X_train)
    X_test_np = col_transformer.transform(X_test)
    
    # getting names for transform data
    # categorical values
    pipe_cats_actual = col_transformer.named_transformers_['cats']
    names_cats = pipe_cats_actual['ohe'].get_feature_names_out()
    #print('Number of categorical names: %d ' %  len(names_cats))
    # numerical values
    names = [name for name, value in num_vars.items() if value]
    names_num = names
    #print('Number of numerical names: %d ' %  len(names_num))
    # lasst of all new names
    names_all = list(names_num) + list(names_cats)

    X_train_df = pd.DataFrame(X_train_np,columns= names_all)
    X_test_df = pd.DataFrame(X_test_np,columns= names_all)
    return X_train_np, X_train_df, X_test_np,X_test_df

#Dictionary of best feature indexes for Logistic regression   
LR_SFS_disct_auc={1:[57],2:[57,52],3:[57,52,72],4:[57,52,72,2],5:[57,52,72,2,15],6:[57,52,72,2,15,32],7:[57,52,72,2,15,32,41],8:[57,52,72,2,15,32,41,73],9:[57,52,72,2,15,32,41,73,17],10:[57,52,72,2,15,32,41,73,17,69],11:[57,52,72,2,15,32,41,73,17,69,46],12:[57,52,72,2,15,32,41,73,17,69,46,30],13:[57,52,72,2,15,32,41,73,17,69,46,30,6],14:[57,52,72,2,15,32,41,73,17,69,46,30,6,19],15:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66],16:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50],17:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11],18:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54],19:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63],20:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71],21:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12],22:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27],23:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40],24:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43],25:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43,3],26:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43,3,65],27:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43,3,65,64],28:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43,3,65,64,62],29:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43,3,65,64,62,16],30:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43,3,65,64,62,16,14],31:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43,3,65,64,62,16,14,48],32:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43,3,65,64,62,16,14,48,47],33:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43,3,65,64,62,16,14,48,47,55],34:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43,3,65,64,62,16,14,48,47,55,25],35:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43,3,65,64,62,16,14,48,47,55,25,51],36:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43,3,65,64,62,16,14,48,47,55,25,51,42],37:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43,3,65,64,62,16,14,48,47,55,25,51,42,45],38:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43,3,65,64,62,16,14,48,47,55,25,51,42,45,38],39:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43,3,65,64,62,16,14,48,47,55,25,51,42,45,38,36],40:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43,3,65,64,62,16,14,48,47,55,25,51,42,45,38,36,60],41:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43,3,65,64,62,16,14,48,47,55,25,51,42,45,38,36,60,61],42:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43,3,65,64,62,16,14,48,47,55,25,51,42,45,38,36,60,61,39],43:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43,3,65,64,62,16,14,48,47,55,25,51,42,45,38,36,60,61,39,35],44:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43,3,65,64,62,16,14,48,47,55,25,51,42,45,38,36,60,61,39,35,34],45:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43,3,65,64,62,16,14,48,47,55,25,51,42,45,38,36,60,61,39,35,34,29],46:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43,3,65,64,62,16,14,48,47,55,25,51,42,45,38,36,60,61,39,35,34,29,21],47:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43,3,65,64,62,16,14,48,47,55,25,51,42,45,38,36,60,61,39,35,34,29,21,23],48:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43,3,65,64,62,16,14,48,47,55,25,51,42,45,38,36,60,61,39,35,34,29,21,23,37],49:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43,3,65,64,62,16,14,48,47,55,25,51,42,45,38,36,60,61,39,35,34,29,21,23,37,26],50:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43,3,65,64,62,16,14,48,47,55,25,51,42,45,38,36,60,61,39,35,34,29,21,23,37,26,9],51:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43,3,65,64,62,16,14,48,47,55,25,51,42,45,38,36,60,61,39,35,34,29,21,23,37,26,9,0],52:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43,3,65,64,62,16,14,48,47,55,25,51,42,45,38,36,60,61,39,35,34,29,21,23,37,26,9,0,13],53:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43,3,65,64,62,16,14,48,47,55,25,51,42,45,38,36,60,61,39,35,34,29,21,23,37,26,9,0,13,8],54:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43,3,65,64,62,16,14,48,47,55,25,51,42,45,38,36,60,61,39,35,34,29,21,23,37,26,9,0,13,8,4],55:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43,3,65,64,62,16,14,48,47,55,25,51,42,45,38,36,60,61,39,35,34,29,21,23,37,26,9,0,13,8,4,44],56:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43,3,65,64,62,16,14,48,47,55,25,51,42,45,38,36,60,61,39,35,34,29,21,23,37,26,9,0,13,8,4,44,56],57:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43,3,65,64,62,16,14,48,47,55,25,51,42,45,38,36,60,61,39,35,34,29,21,23,37,26,9,0,13,8,4,44,56,5],58:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43,3,65,64,62,16,14,48,47,55,25,51,42,45,38,36,60,61,39,35,34,29,21,23,37,26,9,0,13,8,4,44,56,5,67],59:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43,3,65,64,62,16,14,48,47,55,25,51,42,45,38,36,60,61,39,35,34,29,21,23,37,26,9,0,13,8,4,44,56,5,67,1],60:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43,3,65,64,62,16,14,48,47,55,25,51,42,45,38,36,60,61,39,35,34,29,21,23,37,26,9,0,13,8,4,44,56,5,67,1,18],61:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43,3,65,64,62,16,14,48,47,55,25,51,42,45,38,36,60,61,39,35,34,29,21,23,37,26,9,0,13,8,4,44,56,5,67,1,18,7],62:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43,3,65,64,62,16,14,48,47,55,25,51,42,45,38,36,60,61,39,35,34,29,21,23,37,26,9,0,13,8,4,44,56,5,67,1,18,7,22],63:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43,3,65,64,62,16,14,48,47,55,25,51,42,45,38,36,60,61,39,35,34,29,21,23,37,26,9,0,13,8,4,44,56,5,67,1,18,7,22,20],64:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43,3,65,64,62,16,14,48,47,55,25,51,42,45,38,36,60,61,39,35,34,29,21,23,37,26,9,0,13,8,4,44,56,5,67,1,18,7,22,20,28],65:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43,3,65,64,62,16,14,48,47,55,25,51,42,45,38,36,60,61,39,35,34,29,21,23,37,26,9,0,13,8,4,44,56,5,67,1,18,7,22,20,28,24],66:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43,3,65,64,62,16,14,48,47,55,25,51,42,45,38,36,60,61,39,35,34,29,21,23,37,26,9,0,13,8,4,44,56,5,67,1,18,7,22,20,28,24,31],67:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43,3,65,64,62,16,14,48,47,55,25,51,42,45,38,36,60,61,39,35,34,29,21,23,37,26,9,0,13,8,4,44,56,5,67,1,18,7,22,20,28,24,31,10],68:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43,3,65,64,62,16,14,48,47,55,25,51,42,45,38,36,60,61,39,35,34,29,21,23,37,26,9,0,13,8,4,44,56,5,67,1,18,7,22,20,28,24,31,10,70],69:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43,3,65,64,62,16,14,48,47,55,25,51,42,45,38,36,60,61,39,35,34,29,21,23,37,26,9,0,13,8,4,44,56,5,67,1,18,7,22,20,28,24,31,10,70,58],70:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43,3,65,64,62,16,14,48,47,55,25,51,42,45,38,36,60,61,39,35,34,29,21,23,37,26,9,0,13,8,4,44,56,5,67,1,18,7,22,20,28,24,31,10,70,58,68],71:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43,3,65,64,62,16,14,48,47,55,25,51,42,45,38,36,60,61,39,35,34,29,21,23,37,26,9,0,13,8,4,44,56,5,67,1,18,7,22,20,28,24,31,10,70,58,68,59],72:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43,3,65,64,62,16,14,48,47,55,25,51,42,45,38,36,60,61,39,35,34,29,21,23,37,26,9,0,13,8,4,44,56,5,67,1,18,7,22,20,28,24,31,10,70,58,68,59,49],73:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43,3,65,64,62,16,14,48,47,55,25,51,42,45,38,36,60,61,39,35,34,29,21,23,37,26,9,0,13,8,4,44,56,5,67,1,18,7,22,20,28,24,31,10,70,58,68,59,49,33],74:[57,52,72,2,15,32,41,73,17,69,46,30,6,19,66,50,11,54,63,71,12,27,40,43,3,65,64,62,16,14,48,47,55,25,51,42,45,38,36,60,61,39,35,34,29,21,23,37,26,9,0,13,8,4,44,56,5,67,1,18,7,22,20,28,24,31,10,70,58,68,59,49,33,53]}

#Dictionary of best feature indexes for Naive Bayes 
NB_SFS_disct_auc={1:[57],2:[57,52],3:[57,52,32],4:[57,52,32,72],5:[57,52,32,72,73],6:[57,52,32,72,73,21],7:[57,52,32,72,73,21,33],8:[57,52,32,72,73,21,33,2],9:[57,52,32,72,73,21,33,2,36],10:[57,52,32,72,73,21,33,2,36,66],11:[57,52,32,72,73,21,33,2,36,66,17],12:[57,52,32,72,73,21,33,2,36,66,17,6],13:[57,52,32,72,73,21,33,2,36,66,17,6,11],14:[57,52,32,72,73,21,33,2,36,66,17,6,11,26],15:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68],16:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67],17:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16],18:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15],19:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30],20:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65],21:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8],22:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53],23:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27],24:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40],25:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40,42],26:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40,42,43],27:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40,42,43,71],28:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40,42,43,71,18],29:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40,42,43,71,18,62],30:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40,42,43,71,18,62,63],31:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40,42,43,71,18,62,63,12],32:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40,42,43,71,18,62,63,12,44],33:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40,42,43,71,18,62,63,12,44,13],34:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40,42,43,71,18,62,63,12,44,13,19],35:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40,42,43,71,18,62,63,12,44,13,19,56],36:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40,42,43,71,18,62,63,12,44,13,19,56,47],37:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40,42,43,71,18,62,63,12,44,13,19,56,47,60],38:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40,42,43,71,18,62,63,12,44,13,19,56,47,60,14],39:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40,42,43,71,18,62,63,12,44,13,19,56,47,60,14,23],40:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40,42,43,71,18,62,63,12,44,13,19,56,47,60,14,23,7],41:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40,42,43,71,18,62,63,12,44,13,19,56,47,60,14,23,7,41],42:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40,42,43,71,18,62,63,12,44,13,19,56,47,60,14,23,7,41,9],43:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40,42,43,71,18,62,63,12,44,13,19,56,47,60,14,23,7,41,9,34],44:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40,42,43,71,18,62,63,12,44,13,19,56,47,60,14,23,7,41,9,34,25],45:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40,42,43,71,18,62,63,12,44,13,19,56,47,60,14,23,7,41,9,34,25,39],46:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40,42,43,71,18,62,63,12,44,13,19,56,47,60,14,23,7,41,9,34,25,39,64],47:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40,42,43,71,18,62,63,12,44,13,19,56,47,60,14,23,7,41,9,34,25,39,64,50],48:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40,42,43,71,18,62,63,12,44,13,19,56,47,60,14,23,7,41,9,34,25,39,64,50,4],49:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40,42,43,71,18,62,63,12,44,13,19,56,47,60,14,23,7,41,9,34,25,39,64,50,4,51],50:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40,42,43,71,18,62,63,12,44,13,19,56,47,60,14,23,7,41,9,34,25,39,64,50,4,51,61],51:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40,42,43,71,18,62,63,12,44,13,19,56,47,60,14,23,7,41,9,34,25,39,64,50,4,51,61,38],52:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40,42,43,71,18,62,63,12,44,13,19,56,47,60,14,23,7,41,9,34,25,39,64,50,4,51,61,38,35],53:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40,42,43,71,18,62,63,12,44,13,19,56,47,60,14,23,7,41,9,34,25,39,64,50,4,51,61,38,35,0],54:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40,42,43,71,18,62,63,12,44,13,19,56,47,60,14,23,7,41,9,34,25,39,64,50,4,51,61,38,35,0,28],55:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40,42,43,71,18,62,63,12,44,13,19,56,47,60,14,23,7,41,9,34,25,39,64,50,4,51,61,38,35,0,28,1],56:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40,42,43,71,18,62,63,12,44,13,19,56,47,60,14,23,7,41,9,34,25,39,64,50,4,51,61,38,35,0,28,1,3],57:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40,42,43,71,18,62,63,12,44,13,19,56,47,60,14,23,7,41,9,34,25,39,64,50,4,51,61,38,35,0,28,1,3,49],58:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40,42,43,71,18,62,63,12,44,13,19,56,47,60,14,23,7,41,9,34,25,39,64,50,4,51,61,38,35,0,28,1,3,49,37],59:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40,42,43,71,18,62,63,12,44,13,19,56,47,60,14,23,7,41,9,34,25,39,64,50,4,51,61,38,35,0,28,1,3,49,37,5],60:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40,42,43,71,18,62,63,12,44,13,19,56,47,60,14,23,7,41,9,34,25,39,64,50,4,51,61,38,35,0,28,1,3,49,37,5,46],61:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40,42,43,71,18,62,63,12,44,13,19,56,47,60,14,23,7,41,9,34,25,39,64,50,4,51,61,38,35,0,28,1,3,49,37,5,46,69],62:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40,42,43,71,18,62,63,12,44,13,19,56,47,60,14,23,7,41,9,34,25,39,64,50,4,51,61,38,35,0,28,1,3,49,37,5,46,69,20],63:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40,42,43,71,18,62,63,12,44,13,19,56,47,60,14,23,7,41,9,34,25,39,64,50,4,51,61,38,35,0,28,1,3,49,37,5,46,69,20,45],64:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40,42,43,71,18,62,63,12,44,13,19,56,47,60,14,23,7,41,9,34,25,39,64,50,4,51,61,38,35,0,28,1,3,49,37,5,46,69,20,45,22],65:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40,42,43,71,18,62,63,12,44,13,19,56,47,60,14,23,7,41,9,34,25,39,64,50,4,51,61,38,35,0,28,1,3,49,37,5,46,69,20,45,22,48],66:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40,42,43,71,18,62,63,12,44,13,19,56,47,60,14,23,7,41,9,34,25,39,64,50,4,51,61,38,35,0,28,1,3,49,37,5,46,69,20,45,22,48,29],67:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40,42,43,71,18,62,63,12,44,13,19,56,47,60,14,23,7,41,9,34,25,39,64,50,4,51,61,38,35,0,28,1,3,49,37,5,46,69,20,45,22,48,29,31],68:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40,42,43,71,18,62,63,12,44,13,19,56,47,60,14,23,7,41,9,34,25,39,64,50,4,51,61,38,35,0,28,1,3,49,37,5,46,69,20,45,22,48,29,31,54],69:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40,42,43,71,18,62,63,12,44,13,19,56,47,60,14,23,7,41,9,34,25,39,64,50,4,51,61,38,35,0,28,1,3,49,37,5,46,69,20,45,22,48,29,31,54,55],70:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40,42,43,71,18,62,63,12,44,13,19,56,47,60,14,23,7,41,9,34,25,39,64,50,4,51,61,38,35,0,28,1,3,49,37,5,46,69,20,45,22,48,29,31,54,55,70],71:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40,42,43,71,18,62,63,12,44,13,19,56,47,60,14,23,7,41,9,34,25,39,64,50,4,51,61,38,35,0,28,1,3,49,37,5,46,69,20,45,22,48,29,31,54,55,70,10],72:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40,42,43,71,18,62,63,12,44,13,19,56,47,60,14,23,7,41,9,34,25,39,64,50,4,51,61,38,35,0,28,1,3,49,37,5,46,69,20,45,22,48,29,31,54,55,70,10,24],73:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40,42,43,71,18,62,63,12,44,13,19,56,47,60,14,23,7,41,9,34,25,39,64,50,4,51,61,38,35,0,28,1,3,49,37,5,46,69,20,45,22,48,29,31,54,55,70,10,24,58],74:[57,52,32,72,73,21,33,2,36,66,17,6,11,26,68,67,16,15,30,65,8,53,27,40,42,43,71,18,62,63,12,44,13,19,56,47,60,14,23,7,41,9,34,25,39,64,50,4,51,61,38,35,0,28,1,3,49,37,5,46,69,20,45,22,48,29,31,54,55,70,10,24,58,59]}

#Dictionary of best feature indexes for Decision Tree
DT_SFS_disct_auc={1:[57],2:[57,72],3:[57,72,17],4:[57,72,17,67],5:[57,72,17,67,21],6:[57,72,17,67,21,11],7:[57,72,17,67,21,11,73],8:[57,72,17,67,21,11,73,6],9:[57,72,17,67,21,11,73,6,8],10:[57,72,17,67,21,11,73,6,8,26],11:[57,72,17,67,21,11,73,6,8,26,27],12:[57,72,17,67,21,11,73,6,8,26,27,23],13:[57,72,17,67,21,11,73,6,8,26,27,23,41],14:[57,72,17,67,21,11,73,6,8,26,27,23,41,25],15:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51],16:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40],17:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5],18:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43],19:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71],20:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53],21:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18],22:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20],23:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22],24:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42],25:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42,63],26:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42,63,47],27:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42,63,47,19],28:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42,63,47,19,24],29:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42,63,47,19,24,65],30:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42,63,47,19,24,65,55],31:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42,63,47,19,24,65,55,48],32:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42,63,47,19,24,65,55,48,12],33:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42,63,47,19,24,65,55,48,12,45],34:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42,63,47,19,24,65,55,48,12,45,64],35:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42,63,47,19,24,65,55,48,12,45,64,44],36:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42,63,47,19,24,65,55,48,12,45,64,44,56],37:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42,63,47,19,24,65,55,48,12,45,64,44,56,66],38:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42,63,47,19,24,65,55,48,12,45,64,44,56,66,4],39:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42,63,47,19,24,65,55,48,12,45,64,44,56,66,4,62],40:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42,63,47,19,24,65,55,48,12,45,64,44,56,66,4,62,13],41:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42,63,47,19,24,65,55,48,12,45,64,44,56,66,4,62,13,68],42:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42,63,47,19,24,65,55,48,12,45,64,44,56,66,4,62,13,68,70],43:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42,63,47,19,24,65,55,48,12,45,64,44,56,66,4,62,13,68,70,61],44:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42,63,47,19,24,65,55,48,12,45,64,44,56,66,4,62,13,68,70,61,50],45:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42,63,47,19,24,65,55,48,12,45,64,44,56,66,4,62,13,68,70,61,50,49],46:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42,63,47,19,24,65,55,48,12,45,64,44,56,66,4,62,13,68,70,61,50,49,3],47:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42,63,47,19,24,65,55,48,12,45,64,44,56,66,4,62,13,68,70,61,50,49,3,15],48:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42,63,47,19,24,65,55,48,12,45,64,44,56,66,4,62,13,68,70,61,50,49,3,15,10],49:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42,63,47,19,24,65,55,48,12,45,64,44,56,66,4,62,13,68,70,61,50,49,3,15,10,2],50:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42,63,47,19,24,65,55,48,12,45,64,44,56,66,4,62,13,68,70,61,50,49,3,15,10,2,60],51:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42,63,47,19,24,65,55,48,12,45,64,44,56,66,4,62,13,68,70,61,50,49,3,15,10,2,60,54],52:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42,63,47,19,24,65,55,48,12,45,64,44,56,66,4,62,13,68,70,61,50,49,3,15,10,2,60,54,1],53:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42,63,47,19,24,65,55,48,12,45,64,44,56,66,4,62,13,68,70,61,50,49,3,15,10,2,60,54,1,59],54:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42,63,47,19,24,65,55,48,12,45,64,44,56,66,4,62,13,68,70,61,50,49,3,15,10,2,60,54,1,59,46],55:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42,63,47,19,24,65,55,48,12,45,64,44,56,66,4,62,13,68,70,61,50,49,3,15,10,2,60,54,1,59,46,58],56:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42,63,47,19,24,65,55,48,12,45,64,44,56,66,4,62,13,68,70,61,50,49,3,15,10,2,60,54,1,59,46,58,69],57:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42,63,47,19,24,65,55,48,12,45,64,44,56,66,4,62,13,68,70,61,50,49,3,15,10,2,60,54,1,59,46,58,69,14],58:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42,63,47,19,24,65,55,48,12,45,64,44,56,66,4,62,13,68,70,61,50,49,3,15,10,2,60,54,1,59,46,58,69,14,28],59:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42,63,47,19,24,65,55,48,12,45,64,44,56,66,4,62,13,68,70,61,50,49,3,15,10,2,60,54,1,59,46,58,69,14,28,7],60:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42,63,47,19,24,65,55,48,12,45,64,44,56,66,4,62,13,68,70,61,50,49,3,15,10,2,60,54,1,59,46,58,69,14,28,7,16],61:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42,63,47,19,24,65,55,48,12,45,64,44,56,66,4,62,13,68,70,61,50,49,3,15,10,2,60,54,1,59,46,58,69,14,28,7,16,9],62:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42,63,47,19,24,65,55,48,12,45,64,44,56,66,4,62,13,68,70,61,50,49,3,15,10,2,60,54,1,59,46,58,69,14,28,7,16,9,0],63:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42,63,47,19,24,65,55,48,12,45,64,44,56,66,4,62,13,68,70,61,50,49,3,15,10,2,60,54,1,59,46,58,69,14,28,7,16,9,0,34],64:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42,63,47,19,24,65,55,48,12,45,64,44,56,66,4,62,13,68,70,61,50,49,3,15,10,2,60,54,1,59,46,58,69,14,28,7,16,9,0,34,30],65:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42,63,47,19,24,65,55,48,12,45,64,44,56,66,4,62,13,68,70,61,50,49,3,15,10,2,60,54,1,59,46,58,69,14,28,7,16,9,0,34,30,37],66:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42,63,47,19,24,65,55,48,12,45,64,44,56,66,4,62,13,68,70,61,50,49,3,15,10,2,60,54,1,59,46,58,69,14,28,7,16,9,0,34,30,37,38],67:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42,63,47,19,24,65,55,48,12,45,64,44,56,66,4,62,13,68,70,61,50,49,3,15,10,2,60,54,1,59,46,58,69,14,28,7,16,9,0,34,30,37,38,36],68:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42,63,47,19,24,65,55,48,12,45,64,44,56,66,4,62,13,68,70,61,50,49,3,15,10,2,60,54,1,59,46,58,69,14,28,7,16,9,0,34,30,37,38,36,32],69:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42,63,47,19,24,65,55,48,12,45,64,44,56,66,4,62,13,68,70,61,50,49,3,15,10,2,60,54,1,59,46,58,69,14,28,7,16,9,0,34,30,37,38,36,32,35],70:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42,63,47,19,24,65,55,48,12,45,64,44,56,66,4,62,13,68,70,61,50,49,3,15,10,2,60,54,1,59,46,58,69,14,28,7,16,9,0,34,30,37,38,36,32,35,33],71:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42,63,47,19,24,65,55,48,12,45,64,44,56,66,4,62,13,68,70,61,50,49,3,15,10,2,60,54,1,59,46,58,69,14,28,7,16,9,0,34,30,37,38,36,32,35,33,52],72:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42,63,47,19,24,65,55,48,12,45,64,44,56,66,4,62,13,68,70,61,50,49,3,15,10,2,60,54,1,59,46,58,69,14,28,7,16,9,0,34,30,37,38,36,32,35,33,52,39],73:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42,63,47,19,24,65,55,48,12,45,64,44,56,66,4,62,13,68,70,61,50,49,3,15,10,2,60,54,1,59,46,58,69,14,28,7,16,9,0,34,30,37,38,36,32,35,33,52,39,29],74:[57,72,17,67,21,11,73,6,8,26,27,23,41,25,51,40,5,43,71,53,18,20,22,42,63,47,19,24,65,55,48,12,45,64,44,56,66,4,62,13,68,70,61,50,49,3,15,10,2,60,54,1,59,46,58,69,14,28,7,16,9,0,34,30,37,38,36,32,35,33,52,39,29,31]}

LR_SFS_disct_mcc={1:[57],2:[57,1],3:[57,1,2],4:[57,1,2,3],5:[57,1,2,3,4],6:[57,1,2,3,4,9],7:[57,1,2,3,4,9,6],8:[57,1,2,3,4,9,6,8],9:[57,1,2,3,4,9,6,8,13],10:[57,1,2,3,4,9,6,8,13,14],11:[57,1,2,3,4,9,6,8,13,14,15],12:[57,1,2,3,4,9,6,8,13,14,15,7],13:[57,1,2,3,4,9,6,8,13,14,15,7,16],14:[57,1,2,3,4,9,6,8,13,14,15,7,16,33],15:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67],16:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17],17:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12],18:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18],19:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19],20:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23],21:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25],22:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27],23:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31],24:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28],25:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28,30],26:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28,30,34],27:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28,30,34,40],28:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28,30,34,40,42],29:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28,30,34,40,42,43],30:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28,30,34,40,42,43,44],31:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28,30,34,40,42,43,44,45],32:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28,30,34,40,42,43,44,45,46],33:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28,30,34,40,42,43,44,45,46,48],34:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28,30,34,40,42,43,44,45,46,48,56],35:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28,30,34,40,42,43,44,45,46,48,56,63],36:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28,30,34,40,42,43,44,45,46,48,56,63,71],37:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28,30,34,40,42,43,44,45,46,48,56,63,71,49],38:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28,30,34,40,42,43,44,45,46,48,56,63,71,49,39],39:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28,30,34,40,42,43,44,45,46,48,56,63,71,49,39,51],40:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28,30,34,40,42,43,44,45,46,48,56,63,71,49,39,51,53],41:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28,30,34,40,42,43,44,45,46,48,56,63,71,49,39,51,53,54],42:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28,30,34,40,42,43,44,45,46,48,56,63,71,49,39,51,53,54,38],43:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28,30,34,40,42,43,44,45,46,48,56,63,71,49,39,51,53,54,38,60],44:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28,30,34,40,42,43,44,45,46,48,56,63,71,49,39,51,53,54,38,60,11],45:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28,30,34,40,42,43,44,45,46,48,56,63,71,49,39,51,53,54,38,60,11,55],46:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28,30,34,40,42,43,44,45,46,48,56,63,71,49,39,51,53,54,38,60,11,55,61],47:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28,30,34,40,42,43,44,45,46,48,56,63,71,49,39,51,53,54,38,60,11,55,61,22],48:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28,30,34,40,42,43,44,45,46,48,56,63,71,49,39,51,53,54,38,60,11,55,61,22,64],49:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28,30,34,40,42,43,44,45,46,48,56,63,71,49,39,51,53,54,38,60,11,55,61,22,64,62],50:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28,30,34,40,42,43,44,45,46,48,56,63,71,49,39,51,53,54,38,60,11,55,61,22,64,62,65],51:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28,30,34,40,42,43,44,45,46,48,56,63,71,49,39,51,53,54,38,60,11,55,61,22,64,62,65,41],52:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28,30,34,40,42,43,44,45,46,48,56,63,71,49,39,51,53,54,38,60,11,55,61,22,64,62,65,41,66],53:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28,30,34,40,42,43,44,45,46,48,56,63,71,49,39,51,53,54,38,60,11,55,61,22,64,62,65,41,66,47],54:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28,30,34,40,42,43,44,45,46,48,56,63,71,49,39,51,53,54,38,60,11,55,61,22,64,62,65,41,66,47,35],55:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28,30,34,40,42,43,44,45,46,48,56,63,71,49,39,51,53,54,38,60,11,55,61,22,64,62,65,41,66,47,35,21],56:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28,30,34,40,42,43,44,45,46,48,56,63,71,49,39,51,53,54,38,60,11,55,61,22,64,62,65,41,66,47,35,21,26],57:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28,30,34,40,42,43,44,45,46,48,56,63,71,49,39,51,53,54,38,60,11,55,61,22,64,62,65,41,66,47,35,21,26,50],58:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28,30,34,40,42,43,44,45,46,48,56,63,71,49,39,51,53,54,38,60,11,55,61,22,64,62,65,41,66,47,35,21,26,50,32],59:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28,30,34,40,42,43,44,45,46,48,56,63,71,49,39,51,53,54,38,60,11,55,61,22,64,62,65,41,66,47,35,21,26,50,32,68],60:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28,30,34,40,42,43,44,45,46,48,56,63,71,49,39,51,53,54,38,60,11,55,61,22,64,62,65,41,66,47,35,21,26,50,32,68,5],61:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28,30,34,40,42,43,44,45,46,48,56,63,71,49,39,51,53,54,38,60,11,55,61,22,64,62,65,41,66,47,35,21,26,50,32,68,5,73],62:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28,30,34,40,42,43,44,45,46,48,56,63,71,49,39,51,53,54,38,60,11,55,61,22,64,62,65,41,66,47,35,21,26,50,32,68,5,73,70],63:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28,30,34,40,42,43,44,45,46,48,56,63,71,49,39,51,53,54,38,60,11,55,61,22,64,62,65,41,66,47,35,21,26,50,32,68,5,73,70,0],64:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28,30,34,40,42,43,44,45,46,48,56,63,71,49,39,51,53,54,38,60,11,55,61,22,64,62,65,41,66,47,35,21,26,50,32,68,5,73,70,0,20],65:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28,30,34,40,42,43,44,45,46,48,56,63,71,49,39,51,53,54,38,60,11,55,61,22,64,62,65,41,66,47,35,21,26,50,32,68,5,73,70,0,20,36],66:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28,30,34,40,42,43,44,45,46,48,56,63,71,49,39,51,53,54,38,60,11,55,61,22,64,62,65,41,66,47,35,21,26,50,32,68,5,73,70,0,20,36,37],67:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28,30,34,40,42,43,44,45,46,48,56,63,71,49,39,51,53,54,38,60,11,55,61,22,64,62,65,41,66,47,35,21,26,50,32,68,5,73,70,0,20,36,37,10],68:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28,30,34,40,42,43,44,45,46,48,56,63,71,49,39,51,53,54,38,60,11,55,61,22,64,62,65,41,66,47,35,21,26,50,32,68,5,73,70,0,20,36,37,10,29],69:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28,30,34,40,42,43,44,45,46,48,56,63,71,49,39,51,53,54,38,60,11,55,61,22,64,62,65,41,66,47,35,21,26,50,32,68,5,73,70,0,20,36,37,10,29,24],70:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28,30,34,40,42,43,44,45,46,48,56,63,71,49,39,51,53,54,38,60,11,55,61,22,64,62,65,41,66,47,35,21,26,50,32,68,5,73,70,0,20,36,37,10,29,24,69],71:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28,30,34,40,42,43,44,45,46,48,56,63,71,49,39,51,53,54,38,60,11,55,61,22,64,62,65,41,66,47,35,21,26,50,32,68,5,73,70,0,20,36,37,10,29,24,69,72],72:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28,30,34,40,42,43,44,45,46,48,56,63,71,49,39,51,53,54,38,60,11,55,61,22,64,62,65,41,66,47,35,21,26,50,32,68,5,73,70,0,20,36,37,10,29,24,69,72,52],73:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28,30,34,40,42,43,44,45,46,48,56,63,71,49,39,51,53,54,38,60,11,55,61,22,64,62,65,41,66,47,35,21,26,50,32,68,5,73,70,0,20,36,37,10,29,24,69,72,52,58],74:[57,1,2,3,4,9,6,8,13,14,15,7,16,33,67,17,12,18,19,23,25,27,31,28,30,34,40,42,43,44,45,46,48,56,63,71,49,39,51,53,54,38,60,11,55,61,22,64,62,65,41,66,47,35,21,26,50,32,68,5,73,70,0,20,36,37,10,29,24,69,72,52,58,59]}

NB_SFS_disct_mcc={1:[57],2:[57,0],3:[57,0,1],4:[57,0,1,41],5:[57,0,1,41,3],6:[57,0,1,41,3,6],7:[57,0,1,41,3,6,7],8:[57,0,1,41,3,6,7,8],9:[57,0,1,41,3,6,7,8,9],10:[57,0,1,41,3,6,7,8,9,11],11:[57,0,1,41,3,6,7,8,9,11,38],12:[57,0,1,41,3,6,7,8,9,11,38,12],13:[57,0,1,41,3,6,7,8,9,11,38,12,14],14:[57,0,1,41,3,6,7,8,9,11,38,12,14,16],15:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17],16:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13],17:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18],18:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19],19:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22],20:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55],21:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20],22:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64],23:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27],24:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28],25:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28,33],26:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28,33,36],27:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28,33,36,30],28:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28,33,36,30,39],29:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28,33,36,30,39,44],30:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28,33,36,30,39,44,40],31:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28,33,36,30,39,44,40,42],32:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28,33,36,30,39,44,40,42,43],33:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28,33,36,30,39,44,40,42,43,53],34:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28,33,36,30,39,44,40,42,43,53,56],35:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28,33,36,30,39,44,40,42,43,53,56,62],36:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28,33,36,30,39,44,40,42,43,53,56,62,63],37:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28,33,36,30,39,44,40,42,43,53,56,62,63,71],38:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28,33,36,30,39,44,40,42,43,53,56,62,63,71,25],39:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28,33,36,30,39,44,40,42,43,53,56,62,63,71,25,51],40:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28,33,36,30,39,44,40,42,43,53,56,62,63,71,25,51,45],41:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28,33,36,30,39,44,40,42,43,53,56,62,63,71,25,51,45,68],42:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28,33,36,30,39,44,40,42,43,53,56,62,63,71,25,51,45,68,34],43:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28,33,36,30,39,44,40,42,43,53,56,62,63,71,25,51,45,68,34,2],44:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28,33,36,30,39,44,40,42,43,53,56,62,63,71,25,51,45,68,34,2,23],45:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28,33,36,30,39,44,40,42,43,53,56,62,63,71,25,51,45,68,34,2,23,5],46:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28,33,36,30,39,44,40,42,43,53,56,62,63,71,25,51,45,68,34,2,23,5,48],47:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28,33,36,30,39,44,40,42,43,53,56,62,63,71,25,51,45,68,34,2,23,5,48,65],48:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28,33,36,30,39,44,40,42,43,53,56,62,63,71,25,51,45,68,34,2,23,5,48,65,67],49:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28,33,36,30,39,44,40,42,43,53,56,62,63,71,25,51,45,68,34,2,23,5,48,65,67,54],50:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28,33,36,30,39,44,40,42,43,53,56,62,63,71,25,51,45,68,34,2,23,5,48,65,67,54,4],51:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28,33,36,30,39,44,40,42,43,53,56,62,63,71,25,51,45,68,34,2,23,5,48,65,67,54,4,32],52:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28,33,36,30,39,44,40,42,43,53,56,62,63,71,25,51,45,68,34,2,23,5,48,65,67,54,4,32,35],53:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28,33,36,30,39,44,40,42,43,53,56,62,63,71,25,51,45,68,34,2,23,5,48,65,67,54,4,32,35,29],54:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28,33,36,30,39,44,40,42,43,53,56,62,63,71,25,51,45,68,34,2,23,5,48,65,67,54,4,32,35,29,49],55:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28,33,36,30,39,44,40,42,43,53,56,62,63,71,25,51,45,68,34,2,23,5,48,65,67,54,4,32,35,29,49,15],56:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28,33,36,30,39,44,40,42,43,53,56,62,63,71,25,51,45,68,34,2,23,5,48,65,67,54,4,32,35,29,49,15,37],57:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28,33,36,30,39,44,40,42,43,53,56,62,63,71,25,51,45,68,34,2,23,5,48,65,67,54,4,32,35,29,49,15,37,26],58:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28,33,36,30,39,44,40,42,43,53,56,62,63,71,25,51,45,68,34,2,23,5,48,65,67,54,4,32,35,29,49,15,37,26,47],59:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28,33,36,30,39,44,40,42,43,53,56,62,63,71,25,51,45,68,34,2,23,5,48,65,67,54,4,32,35,29,49,15,37,26,47,60],60:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28,33,36,30,39,44,40,42,43,53,56,62,63,71,25,51,45,68,34,2,23,5,48,65,67,54,4,32,35,29,49,15,37,26,47,60,66],61:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28,33,36,30,39,44,40,42,43,53,56,62,63,71,25,51,45,68,34,2,23,5,48,65,67,54,4,32,35,29,49,15,37,26,47,60,66,61],62:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28,33,36,30,39,44,40,42,43,53,56,62,63,71,25,51,45,68,34,2,23,5,48,65,67,54,4,32,35,29,49,15,37,26,47,60,66,61,31],63:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28,33,36,30,39,44,40,42,43,53,56,62,63,71,25,51,45,68,34,2,23,5,48,65,67,54,4,32,35,29,49,15,37,26,47,60,66,61,31,46],64:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28,33,36,30,39,44,40,42,43,53,56,62,63,71,25,51,45,68,34,2,23,5,48,65,67,54,4,32,35,29,49,15,37,26,47,60,66,61,31,46,70],65:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28,33,36,30,39,44,40,42,43,53,56,62,63,71,25,51,45,68,34,2,23,5,48,65,67,54,4,32,35,29,49,15,37,26,47,60,66,61,31,46,70,50],66:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28,33,36,30,39,44,40,42,43,53,56,62,63,71,25,51,45,68,34,2,23,5,48,65,67,54,4,32,35,29,49,15,37,26,47,60,66,61,31,46,70,50,69],67:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28,33,36,30,39,44,40,42,43,53,56,62,63,71,25,51,45,68,34,2,23,5,48,65,67,54,4,32,35,29,49,15,37,26,47,60,66,61,31,46,70,50,69,21],68:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28,33,36,30,39,44,40,42,43,53,56,62,63,71,25,51,45,68,34,2,23,5,48,65,67,54,4,32,35,29,49,15,37,26,47,60,66,61,31,46,70,50,69,21,72],69:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28,33,36,30,39,44,40,42,43,53,56,62,63,71,25,51,45,68,34,2,23,5,48,65,67,54,4,32,35,29,49,15,37,26,47,60,66,61,31,46,70,50,69,21,72,73],70:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28,33,36,30,39,44,40,42,43,53,56,62,63,71,25,51,45,68,34,2,23,5,48,65,67,54,4,32,35,29,49,15,37,26,47,60,66,61,31,46,70,50,69,21,72,73,10],71:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28,33,36,30,39,44,40,42,43,53,56,62,63,71,25,51,45,68,34,2,23,5,48,65,67,54,4,32,35,29,49,15,37,26,47,60,66,61,31,46,70,50,69,21,72,73,10,52],72:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28,33,36,30,39,44,40,42,43,53,56,62,63,71,25,51,45,68,34,2,23,5,48,65,67,54,4,32,35,29,49,15,37,26,47,60,66,61,31,46,70,50,69,21,72,73,10,52,24],73:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28,33,36,30,39,44,40,42,43,53,56,62,63,71,25,51,45,68,34,2,23,5,48,65,67,54,4,32,35,29,49,15,37,26,47,60,66,61,31,46,70,50,69,21,72,73,10,52,24,58],74:[57,0,1,41,3,6,7,8,9,11,38,12,14,16,17,13,18,19,22,55,20,64,27,28,33,36,30,39,44,40,42,43,53,56,62,63,71,25,51,45,68,34,2,23,5,48,65,67,54,4,32,35,29,49,15,37,26,47,60,66,61,31,46,70,50,69,21,72,73,10,52,24,58,59]}

DT_SFS_disct_mcc={1:[57],2:[57,14],3:[57,14,72],4:[57,14,72,44],5:[57,14,72,44,1],6:[57,14,72,44,1,67],7:[57,14,72,44,1,67,19],8:[57,14,72,44,1,67,19,5],9:[57,14,72,44,1,67,19,5,27],10:[57,14,72,44,1,67,19,5,27,40],11:[57,14,72,44,1,67,19,5,27,40,41],12:[57,14,72,44,1,67,19,5,27,40,41,42],13:[57,14,72,44,1,67,19,5,27,40,41,42,43],14:[57,14,72,44,1,67,19,5,27,40,41,42,43,48],15:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18],16:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51],17:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53],18:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56],19:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71],20:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47],21:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6],22:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23],23:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55],24:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28],25:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28,50],26:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28,50,25],27:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28,50,25,45],28:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28,50,25,45,70],29:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28,50,25,45,70,22],30:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28,50,25,45,70,22,26],31:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28,50,25,45,70,22,26,62],32:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28,50,25,45,70,22,26,62,12],33:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28,50,25,45,70,22,26,62,12,8],34:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28,50,25,45,70,22,26,62,12,8,7],35:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28,50,25,45,70,22,26,62,12,8,7,2],36:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28,50,25,45,70,22,26,62,12,8,7,2,17],37:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28,50,25,45,70,22,26,62,12,8,7,2,17,68],38:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28,50,25,45,70,22,26,62,12,8,7,2,17,68,63],39:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28,50,25,45,70,22,26,62,12,8,7,2,17,68,63,11],40:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28,50,25,45,70,22,26,62,12,8,7,2,17,68,63,11,20],41:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28,50,25,45,70,22,26,62,12,8,7,2,17,68,63,11,20,64],42:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28,50,25,45,70,22,26,62,12,8,7,2,17,68,63,11,20,64,24],43:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28,50,25,45,70,22,26,62,12,8,7,2,17,68,63,11,20,64,24,13],44:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28,50,25,45,70,22,26,62,12,8,7,2,17,68,63,11,20,64,24,13,21],45:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28,50,25,45,70,22,26,62,12,8,7,2,17,68,63,11,20,64,24,13,21,59],46:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28,50,25,45,70,22,26,62,12,8,7,2,17,68,63,11,20,64,24,13,21,59,58],47:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28,50,25,45,70,22,26,62,12,8,7,2,17,68,63,11,20,64,24,13,21,59,58,69],48:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28,50,25,45,70,22,26,62,12,8,7,2,17,68,63,11,20,64,24,13,21,59,58,69,65],49:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28,50,25,45,70,22,26,62,12,8,7,2,17,68,63,11,20,64,24,13,21,59,58,69,65,60],50:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28,50,25,45,70,22,26,62,12,8,7,2,17,68,63,11,20,64,24,13,21,59,58,69,65,60,61],51:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28,50,25,45,70,22,26,62,12,8,7,2,17,68,63,11,20,64,24,13,21,59,58,69,65,60,61,66],52:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28,50,25,45,70,22,26,62,12,8,7,2,17,68,63,11,20,64,24,13,21,59,58,69,65,60,61,66,46],53:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28,50,25,45,70,22,26,62,12,8,7,2,17,68,63,11,20,64,24,13,21,59,58,69,65,60,61,66,46,4],54:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28,50,25,45,70,22,26,62,12,8,7,2,17,68,63,11,20,64,24,13,21,59,58,69,65,60,61,66,46,4,0],55:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28,50,25,45,70,22,26,62,12,8,7,2,17,68,63,11,20,64,24,13,21,59,58,69,65,60,61,66,46,4,0,37],56:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28,50,25,45,70,22,26,62,12,8,7,2,17,68,63,11,20,64,24,13,21,59,58,69,65,60,61,66,46,4,0,37,3],57:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28,50,25,45,70,22,26,62,12,8,7,2,17,68,63,11,20,64,24,13,21,59,58,69,65,60,61,66,46,4,0,37,3,49],58:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28,50,25,45,70,22,26,62,12,8,7,2,17,68,63,11,20,64,24,13,21,59,58,69,65,60,61,66,46,4,0,37,3,49,16],59:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28,50,25,45,70,22,26,62,12,8,7,2,17,68,63,11,20,64,24,13,21,59,58,69,65,60,61,66,46,4,0,37,3,49,16,38],60:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28,50,25,45,70,22,26,62,12,8,7,2,17,68,63,11,20,64,24,13,21,59,58,69,65,60,61,66,46,4,0,37,3,49,16,38,54],61:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28,50,25,45,70,22,26,62,12,8,7,2,17,68,63,11,20,64,24,13,21,59,58,69,65,60,61,66,46,4,0,37,3,49,16,38,54,10],62:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28,50,25,45,70,22,26,62,12,8,7,2,17,68,63,11,20,64,24,13,21,59,58,69,65,60,61,66,46,4,0,37,3,49,16,38,54,10,31],63:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28,50,25,45,70,22,26,62,12,8,7,2,17,68,63,11,20,64,24,13,21,59,58,69,65,60,61,66,46,4,0,37,3,49,16,38,54,10,31,73],64:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28,50,25,45,70,22,26,62,12,8,7,2,17,68,63,11,20,64,24,13,21,59,58,69,65,60,61,66,46,4,0,37,3,49,16,38,54,10,31,73,30],65:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28,50,25,45,70,22,26,62,12,8,7,2,17,68,63,11,20,64,24,13,21,59,58,69,65,60,61,66,46,4,0,37,3,49,16,38,54,10,31,73,30,32],66:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28,50,25,45,70,22,26,62,12,8,7,2,17,68,63,11,20,64,24,13,21,59,58,69,65,60,61,66,46,4,0,37,3,49,16,38,54,10,31,73,30,32,34],67:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28,50,25,45,70,22,26,62,12,8,7,2,17,68,63,11,20,64,24,13,21,59,58,69,65,60,61,66,46,4,0,37,3,49,16,38,54,10,31,73,30,32,34,9],68:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28,50,25,45,70,22,26,62,12,8,7,2,17,68,63,11,20,64,24,13,21,59,58,69,65,60,61,66,46,4,0,37,3,49,16,38,54,10,31,73,30,32,34,9,36],69:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28,50,25,45,70,22,26,62,12,8,7,2,17,68,63,11,20,64,24,13,21,59,58,69,65,60,61,66,46,4,0,37,3,49,16,38,54,10,31,73,30,32,34,9,36,33],70:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28,50,25,45,70,22,26,62,12,8,7,2,17,68,63,11,20,64,24,13,21,59,58,69,65,60,61,66,46,4,0,37,3,49,16,38,54,10,31,73,30,32,34,9,36,33,35],71:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28,50,25,45,70,22,26,62,12,8,7,2,17,68,63,11,20,64,24,13,21,59,58,69,65,60,61,66,46,4,0,37,3,49,16,38,54,10,31,73,30,32,34,9,36,33,35,39],72:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28,50,25,45,70,22,26,62,12,8,7,2,17,68,63,11,20,64,24,13,21,59,58,69,65,60,61,66,46,4,0,37,3,49,16,38,54,10,31,73,30,32,34,9,36,33,35,39,15],73:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28,50,25,45,70,22,26,62,12,8,7,2,17,68,63,11,20,64,24,13,21,59,58,69,65,60,61,66,46,4,0,37,3,49,16,38,54,10,31,73,30,32,34,9,36,33,35,39,15,52],74:[57,14,72,44,1,67,19,5,27,40,41,42,43,48,18,51,53,56,71,47,6,23,55,28,50,25,45,70,22,26,62,12,8,7,2,17,68,63,11,20,64,24,13,21,59,58,69,65,60,61,66,46,4,0,37,3,49,16,38,54,10,31,73,30,32,34,9,36,33,35,39,15,52,29]}

models=[BernoulliNB(), DecisionTreeClassifier(random_state=0,), LogisticRegression(random_state=0, max_iter=300, class_weight='balanced'), KNeighborsClassifier(n_neighbors=6, n_jobs=2)]

#Modeling. Returns: RUC, MCC, CM, Best features selections
def modelling(model,  X_train, X_test, y_train, y_test, eval_strat):
    df_best_features = pd.DataFrame()
        
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    #Metrics for cross validation + blins for all featuers
    cv_roc_auc_mean,cv_mcc_mean,cv_accuracy_mean,cv_f1_mean,cv_precision_mean,cv_recall_mean, roc_auc,mcc,accuracy,f1,precision,recall = return_scores(model,X_train, X_test, y_train, y_test)
    
    if eval_strat=='Cross-validation': 
        create_table_with_metrics_n_column([[cv_roc_auc_mean],[cv_accuracy_mean],[cv_precision_mean],[cv_recall_mean]],model.__class__.__name__,1)   #[cv_mcc_mean],  [cv_f1_mean],

        cv = StratifiedKFold(n_splits=10)
        draw_cv_roc_curve(model, cv, X_train, y_train, title='Cross Validated ROC')
        plt.show()
    
    if eval_strat!='Cross-validation': 
        create_table_with_metrics_n_column([[cv_roc_auc_mean,roc_auc],[cv_accuracy_mean,accuracy],[cv_precision_mean,precision],[cv_recall_mean,recall]],model.__class__.__name__,2) #[cv_mcc_mean,mcc],[cv_f1_mean,f1],
        
        cv = StratifiedKFold(n_splits=10)
        draw_cv_roc_curve(model, cv, X_train, y_train, title='Cross Validated ROC')
        plt.show()
        '''if (model.__class__.__name__=='LogisticRegression' or model.__class__.__name__=='DecisionTreeClassifier'):
            if len(X_test.columns[0:]) > 65:
                fig, ax = plt.subplots(1,3, figsize=(40,5), gridspec_kw={'width_ratios': [1, 1, 4.7]})
            elif len(X_test.columns[0:]) > 45:
                fig, ax = plt.subplots(1,3, figsize=(35,5), gridspec_kw={'width_ratios': [1, 1, 4.3]})
            elif len(X_test.columns[0:]) > 25:
                fig, ax = plt.subplots(1,3, figsize=(30,5), gridspec_kw={'width_ratios': [1, 1, 3.5]})
            elif len(X_test.columns[0:]) > 13:
                fig, ax = plt.subplots(1,3, figsize=(27,5), gridspec_kw={'width_ratios': [1, 1, 2.5]})
            else:
                fig, ax = plt.subplots(1,3, figsize=(25,5), gridspec_kw={'width_ratios': [1, 1, 2]})
        else:'''
        fig, ax = plt.subplots(1,2, figsize=(15,5), gridspec_kw={'width_ratios': [1.5, 1]})
     
        RocCurveDisplay(model, X_test, y_test, ax=ax[0])
        ax[0].plot([0,1], [0,1], linestyle='--', lw=2, label='Chance', alpha=.8)
        
        sns.heatmap(confusion_matrix(y_test, y_pred, labels=[1, 0]), annot=True, fmt='g', ax=ax[1], cmap='Blues',xticklabels=['admitted','non-admitted'],yticklabels=['admitted','non-admitted'])
        ax[1].set_ylabel('Actual')
        ax[1].set_xlabel('Predicted')  
        plt.show()
        
        if model.__class__.__name__ in ('LogisticRegression','DecisionTreeClassifier'):
        #fig, ax = plt.subplots(1,2, figsize=(15,5), gridspec_kw={'width_ratios': [1.5, 1]})
            
            if len(X_test.columns[0:]) > 65:
                fig, ax = plt.subplots(figsize=(35,5))
            elif len(X_test.columns[0:]) > 45:
                fig, ax = plt.subplots(figsize=(25,5))
            elif len(X_test.columns[0:]) > 25:
                fig, ax = plt.subplots(figsize=(18,5))
            else:
                fig, ax = plt.subplots(figsize=(15,5))
        
            list_if_best_features = list(X_test.columns[0:])
            list_if_best_features_renamed = change_categ_features_name(categorical_features_manes2long_names,list_if_best_features)

        #auc_cv=round(np.mean(cross_val_score(model, X_train, y_train, cv=10, scoring='roc_auc')),2)
        
        #print('cv train', auc_cv)
        if model.__class__.__name__=='LogisticRegression':

            feat_imp = {'features': list_if_best_features_renamed, 'coefficients': list(np.exp(model.coef_[0]))}
            df_best_features=pd.DataFrame(feat_imp).sort_values(['coefficients'], ascending=False)
        if model.__class__.__name__=='DecisionTreeClassifier':
            feat_imp = {'features': list_if_best_features_renamed, 'coefficients': list(model.feature_importances_)}
            df_best_features=pd.DataFrame(feat_imp).sort_values(['coefficients'], ascending=False)
        '''if model.__class__.__name__=='BernoulliNB':
            imps = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=0)
            importances = imps.importances_mean
            std = imps.importances_std
            feat_imp={'features': list(X_test.columns[0:]), 'coefficients': list(importances)}
            df_importance=pd.DataFrame(feat_imp).sort_values(['coefficients'], ascending=False)
            df_best_features=df_importance[df_importance.coefficients>0]'''
        if df_best_features.shape[0]>0:   
            plt.bar(df_best_features['features'], df_best_features['coefficients'], align='center', alpha=0.5)
            plt.xticks(rotation=80)
            plt.ylabel('Features importance')
            plt.title('Set of best features for ' + model.__class__.__name__)
            plt.show()
      


def name2model(models_name):
    if models_name=='BernoulliNB':
        model = models[0]
    elif models_name=='DecisionTreeClassifier':
        model = models[1]
    elif models_name=='LogisticRegression':
        model = models[2]
    else:
        model = models[3]
    return model

categorical_features_manes={'hospital_RMH':'x0_RMH', 
'hospital_WH':'x0_WH', 
'sex_F':'x1_F',
'sex_M':'x1_M', 
'smoker_CURRENT': 'x2_CURRENT', 
'smoker_FORMER':'x2_FORMER', 
'smoker_NO':'x2_NO', 
'smoker_UNKNOWN':'x2_UNKNOWN',
'method_BOTH INSULIN AND TABS':'x3_BOTH INSULIN AND TABS', 
'method_DIET ONLY':'x3_DIET ONLY', 
'method_INSULIN AND NON-INSULIN':'x3_INSULIN AND NON-INSULIN', 
'method_INSULIN':'x3_INSULIN',
'method_NON-INSULIN':'x3_NON-INSULIN', 
'method_OTHER':'x3_OTHER',
'method_TABLETS':'x3_TABLETS', 
'method_UNKNOWN':'x3_UNKNOWN'}

categorical_features_manes2long_names={'x0_RMH':'hospital_RMH', 
'x0_WH':'hospital_WH', 
'x1_F':'sex_F',
'x1_M':'sex_M', 
'x2_CURRENT':'smoker_current', 
'x2_FORMER': 'smoker_former', 
'x2_NO':'smoker_no', 
'x2_UNKNOWN':'smoker_unknown',
'x3_BOTH INSULIN AND TABS':'method_both ins and tabs', 
'x3_DIET ONLY':'method_diet only', 
'x3_INSULIN AND NON-INSULIN':'method_ins and non-ins', 
'x3_INSULIN':'method_insulin',
'x3_NON-INSULIN':'method_non-ins', 
'x3_OTHER':'method_other',
'x3_TABLETS':'method_tablets', 
'x3_UNKNOWN':'method_unknown'}

def change_categ_features_name(dictionary, list_of_features):    
    for i in range(len(list_of_features)):
                    if list_of_features[i] in dictionary:
                        list_of_features[i]=dictionary[list_of_features[i]]
    return list_of_features

def checkboxes2features(checkboxes):
    features_list=[]
    for num_goup in range(len(checkboxes)):
        for num_subgr in range(len(checkboxes[num_goup].children)):
            if checkboxes[num_goup].children[num_subgr].value:
                features_list.append(checkboxes[num_goup].children[num_subgr].description)
    for i in range(len(features_list)):
        if features_list[i] in categorical_features_manes:
            features_list[i]=categorical_features_manes[features_list[i]]
            #print(features_list[i])
    return features_list
    
    
# CV plotting AUC    
def draw_cv_roc_curve(classifier, cv, X, y, title='ROC Curve'):
    #fig=plt.figure(figsize=(8, 8))
    fig, ax = plt.subplots(1,2, figsize=(15,5), gridspec_kw={'width_ratios': [1.5, 1]})
    """
    Draw a Cross Validated ROC Curve.
    Args:
        classifier: Classifier Object
        cv: StratifiedKFold Object: 
        X: Feature Pandas DataFrame
        y: Response Pandas Series
    Example largely taken from 
    """
    # Creating ROC Curve with Cross Validation
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    for train, test in cv.split(X, y):
        probas_ = classifier.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y.iloc[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax[0].plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1
    ax[0].plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    
    ax[0].plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax[0].fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    ax[0].set_xlim([-0.05, 1.05])
    ax[0].set_ylim([-0.05, 1.05])
    ax[0].set_xlabel('False Positive Rate')
    ax[0].set_ylabel('True Positive Rate')
    ax[0].set_title(title)
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=2)#loc="lower right"
    
    
    y_pred = cross_val_predict(classifier, X, y, cv=10)
    conf_mat = confusion_matrix(y, y_pred, labels=[1, 0])
    sns.heatmap(conf_mat, annot=True, fmt='g', cmap='Blues', ax=ax[1],xticklabels=['admitted','non-admitted'],yticklabels=['admitted','non-admitted'])
    ax[1].set_ylabel('Actual')
    ax[1].set_xlabel('Predicted')  
    
    plt.show()
    
    
def return_scores(model,  X_train, X_test, y_train, y_test):
    #for CV
    Matthew = make_scorer(matthews_corrcoef)
    cv_roc_auc_mean = round(np.mean(cross_val_score(model, X_train, y_train , cv=10, scoring='roc_auc')),2)
    cv_mcc_mean = round(np.mean(cross_val_score(model, X_train, y_train ,cv=10,scoring=Matthew)),2)    
    cv_accuracy_mean = round(np.mean(cross_val_score(model, X_train, y_train , cv=10, scoring='accuracy')),2)
    cv_f1_mean = round(np.mean(cross_val_score(model, X_train, y_train , cv=10, scoring='f1')),2)
    cv_precision_mean = round(np.mean(cross_val_score(model, X_train, y_train , cv=10, scoring='precision')),2)
    #display(cv_precision_mean)
    cv_recall_mean = round(np.mean(cross_val_score(model, X_train, y_train , cv=10, scoring='recall')),2)
    #for blind test
    roc_auc = round(roc_auc_score(y_test, model.predict_proba(X_test)[:,1]),2)
    y_pred_best = model.predict(X_test)
    mcc = round(matthews_corrcoef(y_test, y_pred_best),2)
    accuracy = round(accuracy_score(y_test, y_pred_best),2)
    f1 = round(f1_score(y_test, y_pred_best, average='binary', zero_division=0),2)
    precision = round(precision_score(y_test, y_pred_best, average='binary', zero_division=0),2)
    recall = round(recall_score(y_test, y_pred_best, average='binary', zero_division=0),2)
    return cv_roc_auc_mean,cv_mcc_mean,cv_accuracy_mean,cv_f1_mean,cv_precision_mean,cv_recall_mean, roc_auc,mcc,accuracy,f1,precision,recall

'''def create_table_with_metrics(metrics_values, name_of_dataframe):
    df = pd.DataFrame(metrics_values,
                      index=pd.Index(['AUC','MCC', 'Accuracy', 'F1', 'Precision', 'Recall'], name='Metrics:'),
                      columns=pd.MultiIndex.from_product([['Model with all available features', 'Model with best combination of features'],['Cross-validation', 'Blind dataset']], names=['Model:', 'Evaluation strategy:']))
    s = df.style.format('{:.2f}')
    cell_hover = {  # for row hover use <tr> instead of <td>
        'selector': 'td:hover',
        'props': [('background-color', '#ffffb3')]
    }
    index_names = {
        'selector': '.index_name',
        'props': 'font-style: italic; color: darkgrey; font-weight:normal;'
    }
    headers = {
        'selector': 'th:not(.index_name)',
        'props': 'background-color: lightblue; color: black;'
    }
    s.set_table_styles([cell_hover, index_names, headers])
    s.set_table_styles([
        {'selector': 'th.col_heading', 'props': 'text-align: center;'},
        {'selector': 'th.col_heading.level0', 'props': 'font-size: 1.5em;'},
        {'selector': 'td', 'props': 'text-align: center; font-weight: bold;'},
    ], overwrite=False)
    s.set_table_styles({
        ('Model with best combination of features', 'Cross-validation'): [{'selector': 'th', 'props': 'border-left: 1px solid white'},
                                   {'selector': 'td', 'props': 'border-left: 1px solid #000066'}]
    }, overwrite=False, axis=0)

    tt = pd.DataFrame([['AUC tells how much a model is capable of distinguishing between classes. The higher the AUC, the better the performance of the model at distinguishing between the positive and negative classes.'],
                       ['The MCC is a correlation coefficient between the observed and predicted binary classifications. It returns a value between −1 and +1, where +1 is a perfect prediction, 0 - random and −1 is total disagreement between prediction and observation. '],
                       ['Accuracy is the fraction of predictions a model got right (Accuracy = Number of correct predictions/total number of predictions).'],
                       ['F1 score is a weighted average of the precision and recall.The result is a value between 0.0 for the worst F-measure and 1.0 for a perfect F-measure.'],
                       ['Precision shows how many selected items are relevant: TP/(TP+FP). A precision score of 1.0 for the class means that every item labelled as belonging to this class.'],
                       ['Recall shows how many relevant items are selected: TP/(TP+FN). The result is a value between 0.0 for no recall and 1.0 for full or perfect recall.']],
                      index=df.index, columns=df.columns[[0]])
    s.set_tooltips(tt, props='visibility: hidden; position: absolute; z-index: 1; border: 1px solid #000066;'
                             'background-color: white; color: #000066; font-size: 0.8em;'
                         'transform: translate(0px, -24px); padding: 0.6em; border-radius: 0.5em;')
    s.set_caption(name_of_dataframe).set_table_styles([{'selector': 'caption', 'props': [('color', 'Navy'),('font-size', '18px'),('font-weight', 'bold'),]}], overwrite=False)
    display(s)'''
    
def create_table_with_metrics_n_column(metrics_values, name_of_dataframe, num_of_column):
    plt.rcParams.update({'font.size': 12})
    if num_of_column==1:
        df = pd.DataFrame(metrics_values,
                  index=pd.Index(['AUC', 'Accuracy',  'Precision', 'Recall(Sensitivity)'], name='Metrics:'), #'MCC','F1',
                  columns=pd.MultiIndex.from_product([['Model with features of your choice'],['Cross-validation']], names=['Model:', 'Evaluation strategy:']))
        tt = pd.DataFrame([['AUC tells how much a model is capable of distinguishing between classes. The higher the AUC, the better the performance of the model at distinguishing between the positive and negative classes. When AUC=0.5, then the classifier is not able to distinguish between Positive and Negative class points.'],
           ['Accuracy is the fraction of predictions a model got right (Accuracy = Number of correct predictions/total number of predictions).'],
           ['Precision shows how many selected items are relevant: TP/(TP+FP). A precision score of 1.0 for the class means that every item labelled as belonging to this class, is correctly assiggned.'],
           ['Recall shows how many relevant items are retrieved: TP/(TP+FN). The result is a value between 0.0 for no recall and 1.0 for perfect recall.']],
          index=df.index, columns=df.columns)
    elif num_of_column==2:
        df = pd.DataFrame(metrics_values,
                  index=pd.Index(['AUC', 'Accuracy', 'Precision', 'Recall(Sensitivity)'], name='Metrics:'), #'MCC','F1', 
                  columns=pd.MultiIndex.from_product([['Model with features of your choice'],['Cross-validation', 'Blind dataset']], names=['Model:', 'Evaluation strategy:']))
                  
        tt = pd.DataFrame([['AUC tells how much a model is capable of distinguishing between classes. The higher the AUC, the better the performance of the model at distinguishing between the positive and negative classes. When AUC=0.5, then the classifier is not able to distinguish between Positive and Negative class points.',
                            'AUC tells how much a model is capable of distinguishing between classes. The higher the AUC, the better the performance of the model at distinguishing between the positive and negative classes. When AUC=0.5, then the classifier is not able to distinguish between Positive and Negative class points.'],
                   ['Accuracy is the fraction of predictions a model got right (Accuracy = Number of correct predictions/total number of predictions).',
                   'Accuracy is the fraction of predictions a model got right (Accuracy = Number of correct predictions/total number of predictions).'],
                   ['Precision shows how many selected items are relevant: TP/(TP+FP). A precision score of 1.0 for the class means that every item labelled as belonging to this class, is correctly assiggned.',
                   'Precision shows how many selected items are relevant: TP/(TP+FP). A precision score of 1.0 for the class means that every item labelled as belonging to this class, is correctly assiggned.'],
                   ['Recall shows how many relevant items are retrieved: TP/(TP+FN). The result is a value between 0.0 for no recall and 1.0 for perfect recall.',
                   'Recall shows how many relevant items are retrieved: TP/(TP+FN). The result is a value between 0.0 for no recall and 1.0 for perfect recall.']],
                  index=df.index, columns=df.columns)
    else:
        df = pd.DataFrame(metrics_values,
                          index=pd.Index(['AUC','Accuracy', 'Precision', 'Recall(Sensitivity)'], name='Metrics:'), #'MCC', 'F1',
                          columns=pd.MultiIndex.from_product([['Model with all available features', 'Model with best combination of features'],['Cross-validation', 'Blind dataset']], names=['Model:', 'Evaluation strategy:']))
                          
        tt = pd.DataFrame([['AUC tells how much a model is capable of distinguishing between classes. The higher the AUC, the better the performance of the model at distinguishing between the positive and negative classes. When AUC=0.5, then the classifier is not able to distinguish between Positive and Negative class points.','AUC tells how much a model is capable of distinguishing between classes. The higher the AUC, the better the performance of the model at distinguishing between the positive and negative classes. When AUC=0.5, then the classifier is not able to distinguish between Positive and Negative class points.','AUC tells how much a model is capable of distinguishing between classes. The higher the AUC, the better the performance of the model at distinguishing between the positive and negative classes. When AUC=0.5, then the classifier is not able to distinguish between Positive and Negative class points.','AUC tells how much a model is capable of distinguishing between classes. The higher the AUC, the better the performance of the model at distinguishing between the positive and negative classes. When AUC=0.5, then the classifier is not able to distinguish between Positive and Negative class points.'],
                   ['Accuracy is the fraction of predictions a model got right (Accuracy = Number of correct predictions/total number of predictions).',
                   'Accuracy is the fraction of predictions a model got right (Accuracy = Number of correct predictions/total number of predictions).',
                   'Accuracy is the fraction of predictions a model got right (Accuracy = Number of correct predictions/total number of predictions).',
                   'Accuracy is the fraction of predictions a model got right (Accuracy = Number of correct predictions/total number of predictions).'],
                   ['Precision shows how many selected items are relevant: TP/(TP+FP). A precision score of 1.0 for the class means that every item labelled as belonging to this class, is correctly assiggned.',
                   'Precision shows how many selected items are relevant: TP/(TP+FP). A precision score of 1.0 for the class means that every item labelled as belonging to this class, is correctly assiggned.',
                   'Precision shows how many selected items are relevant: TP/(TP+FP). A precision score of 1.0 for the class means that every item labelled as belonging to this class, is correctly assiggned.',
                   'Precision shows how many selected items are relevant: TP/(TP+FP). A precision score of 1.0 for the class means that every item labelled as belonging to this class, is correctly assiggned.'],
                   ['Recall shows how many relevant items are retrieved: TP/(TP+FN). The result is a value between 0.0 for no recall and 1.0 for perfect recall.',
                   'Recall shows how many relevant items are retrieved: TP/(TP+FN). The result is a value between 0.0 for no recall and 1.0 for perfect recall.',
                   'Recall shows how many relevant items are retrieved: TP/(TP+FN). The result is a value between 0.0 for no recall and 1.0 for perfect recall.',
                   'Recall shows how many relevant items are retrieved: TP/(TP+FN). The result is a value between 0.0 for no recall and 1.0 for perfect recall.']],
                  index=df.index, columns=df.columns)
    s = df.style.format('{:.2f}')
    cell_hover = {  # for row hover use <tr> instead of <td>
        'selector': 'td:hover',
        'props': [('background-color', '#ffffb3')]
    }
    index_names = {
        'selector': '.index_name',
        'props': 'font-style: italic; color: darkgrey; font-weight:normal;'
    }
    headers = {
        'selector': 'th:not(.index_name)',
        'props': 'background-color: #08306B; color: white;'  #white #EAF3FB
    }
    s.set_table_styles([cell_hover, index_names, headers])
    s.set_table_styles([
        {'selector': 'th.col_heading', 'props': 'text-align: center;'},
        {'selector': 'th.col_heading.level0', 'props': 'font-size: 1.5em;'},
        {'selector': 'td', 'props': 'text-align: center; font-weight: bold;'},
    ], overwrite=False)
    s.set_table_styles({
        ('Model with best combination of features', 'Cross-validation'): [{'selector': 'th', 'props': 'border-left: 1px solid white'},
                                   {'selector': 'td', 'props': 'border-left: 1px solid #000066'}]
    }, overwrite=False, axis=0)


                      
    #['The MCC is a correlation coefficient between the observed and predicted binary classifications. It returns a value between −1 and +1, where +1 is a perfect prediction, 0 - random and −1 is total disagreement between prediction and observation. '],
    #                       ['F1 score is a weighted average of the precision and recall.The result is a value between 0.0 for the worst F-measure and 1.0 for a perfect F-measure.'],
    s.set_tooltips(tt, props='visibility: hidden; position: absolute; z-index: 1; border: 1px solid #000066;'
                             'background-color: white; color: #000066; font-size: 0.8em;'
                         'transform: translate(0px, -24px); padding: 0.6em; border-radius: 0.5em;')
    s.set_caption(name_of_dataframe).set_table_styles([{'selector': 'caption', 'props': [('color', 'Black'),('font-size', '30px'),('font-weight', 'bold'),]}], overwrite=False)
    display(s)
    
def modelling_base_vs_best(model,  X_train, X_test, y_train, y_test):
        
    model.fit(X_train, y_train)    
    y_pred = model.predict(X_test)     
    
    #Metrics for cross validation + blins for all featuers
    cv_roc_auc_mean,cv_mcc_mean,cv_accuracy_mean,cv_f1_mean,cv_precision_mean,cv_recall_mean, roc_auc,mcc,accuracy,f1,precision,recall = return_scores(model,X_train, X_test, y_train, y_test)
    

    #Plot size depends on ML algorithm
    if (model.__class__.__name__=='LogisticRegression' ): #or model.__class__.__name__=='DecisionTreeClassifier'
        #fig, ax = plt.subplots(1,5, figsize=(40,5), gridspec_kw={'width_ratios': [1, 1, 1, 1, 3]})
        fig, ax = plt.subplots(1,4, figsize=(27,5))
        fig.suptitle('MODEL WITH ALL 74 AVAILABLE FEATURES ON BLIND DATASET             MODEL WITH BEST COMBINATION OF 13 FEATURES ON BLIND DATASET', y=1.09, x=0.55, fontsize=21,  fontweight='bold')
        
    else:
        fig, ax = plt.subplots(1,4, figsize=(27,5))
        fig.suptitle('MODEL WITH ALL 74 AVAILABLE FEATURES ON BLIND DATASET             MODEL WITH BEST COMBINATION OF 10 FEATURES ON BLIND DATASET', y=1.09, x=0.55, fontsize=21,  fontweight='bold')
    #base model
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax[0])
    ax[0].plot([0,1], [0,1], linestyle='--', lw=2, label='Chance', alpha=.8)
    sns.heatmap(confusion_matrix(y_test, y_pred, labels=[1, 0]), annot=True, fmt='g', ax=ax[1], cmap='Blues',xticklabels=['admitted','non-admitted'],yticklabels=['admitted','non-admitted'])   
    ax[1].set_ylabel('Actual')
    ax[1].set_xlabel('Predicted')  
    
    #Create datasets (train/test) with best features
    if model.__class__.__name__=='LogisticRegression':
        X_train_best = X_train.iloc[:, LR_SFS_disct_auc[13]]
        X_test_best = X_test.iloc[:, LR_SFS_disct_auc[13]]
    elif model.__class__.__name__=='BernoulliNB':
        X_train_best = X_train.iloc[:, NB_SFS_disct_auc[10]]
        X_test_best = X_test.iloc[:, NB_SFS_disct_auc[10]]
    else:
        X_train_best = X_train.iloc[:, DT_SFS_disct_auc[10]]
        X_test_best = X_test.iloc[:, DT_SFS_disct_auc[10]]        
  
    
    model.fit(X_train_best, y_train)
    y_pred_best = model.predict(X_test_best)   
    #return metrics for best sets of features
    cv_roc_auc_mean_best,cv_mcc_mean_best,cv_accuracy_mean_best,cv_f1_mean_best,cv_precision_mean_best,cv_recall_mean_best, roc_auc_best, mcc_best, accuracy_best, f1_best, precision_best, recall_best = return_scores(model,X_train_best, X_test_best, y_train, y_test)
    
    #Metrics for cv and blind set
    create_table_with_metrics_n_column([[cv_roc_auc_mean,roc_auc,cv_roc_auc_mean_best,roc_auc_best],[cv_accuracy_mean,accuracy,cv_accuracy_mean_best,accuracy_best],[cv_precision_mean,precision,cv_precision_mean_best,precision_best], [cv_recall_mean,recall,cv_recall_mean_best,recall_best]],model.__class__.__name__,4)
    #[cv_mcc_mean,mcc,cv_mcc_mean_best,mcc_best],[cv_f1_mean,f1,cv_f1_mean_best,f1_best],
    #plotting AUC
    RocCurveDisplay.from_estimator(model, X_test_best, y_test, ax=ax[2])
    ax[2].plot([0,1], [0,1], linestyle='--', lw=2, label='Chance', alpha=.8)
    #Heatmap for CV
    sns.heatmap(confusion_matrix(y_test, y_pred_best, labels=[1, 0]), annot=True, fmt='g', ax=ax[3], cmap='Oranges',xticklabels=['admitted','non-admitted'],yticklabels=['admitted','non-admitted'])  
    ax[3].set_ylabel('Actual')
    ax[3].set_xlabel('Predicted')     
    
    plt.show()   
    #Return best features with coiffecients
    df_best_features=pd.DataFrame()
    
    if model.__class__.__name__ in ('LogisticRegression','DecisionTreeClassifier'):
        fig, ax = plt.subplots(figsize=(15,5))
        list_if_best_features = list(X_test_best.columns[0:])
        list_if_best_features_renamed = change_categ_features_name(categorical_features_manes2long_names,list_if_best_features)
            
    #For best model only
    if model.__class__.__name__=='LogisticRegression':
        feat_imp = {'features': list_if_best_features_renamed, 'coefficients': list(np.exp(model.coef_[0]))}
        df_best_features=pd.DataFrame(feat_imp).sort_values(['coefficients'], ascending=False)
        #fig.suptitle('MODEL WITH ALL 74 AVAILABLE FEATURES ON BLIND DATASET                MODEL WITH BEST COMBINATION OF 13 FEATURES ON BLIND DATASET', y=1.09, x=0.4, fontsize=21,  fontweight='bold')
    if model.__class__.__name__=='DecisionTreeClassifier':
        feat_imp = {'features': list_if_best_features_renamed, 'coefficients': list(model.feature_importances_)}
        df_best_features=pd.DataFrame(feat_imp).sort_values(['coefficients'], ascending=False)
        
    '''if model.__class__.__name__=='BernoulliNB':
        imps = permutation_importance(model, X_test_best, y_test, n_repeats=30, random_state=0)
        importances = imps.importances_mean
        std = imps.importances_std
        feat_imp={'features': list(X_test_best.columns[0:]), 'coefficients': list(importances)}
        df_importance=pd.DataFrame(feat_imp).sort_values(['coefficients'], ascending=False)
        df_best_features=df_importance[df_importance.coefficients>0]'''
    if df_best_features.shape[0]>0:   
        plt.bar(df_best_features['features'], df_best_features['coefficients'], align='center', alpha=0.5, width=0.9)
        plt.ylabel('Features importance')
        plt.xticks(rotation=80)
        plt.title('Set of best features for ' + model.__class__.__name__)
    
        plt.show()
