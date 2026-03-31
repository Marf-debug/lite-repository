import numpy as np
import pandas as pd
from operator import eq, ge, gt, ne, lt, le
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *
from time import time
# Pretty plots
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')
plt.rcParams['figure.figsize'] = (6, 4)
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.title_fontsize'] = 12
plt.rcParams['legend.fontsize'] = 12


COLUMN_DEFINITIONS = {
    'patient_id': "Patient's identifier",
    'hospital': "Hospital where the patient was admitted: RMH, WH",
    # Demographics
    'age': "Age as of 01/01/2025, all patients aged 86+ were grouped into one category '86'",
    'sex': "Column sex has 2 categories: M, F",
    'height': "Patient's height (cm)",
    'weight': "Patient's weight at admission (kg)",
    'weight_over_100': "Whether the patient is over 100kg",
    'bmi': "Patient's body mass index",
    'waist': "Patient's waist circumference (cm)",
    'obesity': "Flag for whether the patient is obese",
    'interpreter': "Whether the patient requires an interpreter",
    'alcohol': "Whether the person drinks alcohol",
    'smoker': "Smoking status of the person",
    # Vitals
    'systolic_bp': "The person's systolic blood pressure",
    'diastolic_bp': "The person's diastolic blood pressure",
    'elevated_bp': "Indicates whether the person has elevated blood pressure ( > 190/95 mmHg)",
    # Diabetes history
    'years_since_diagnosis': "Number of years since the person has been diagnosed with diabetes type 2 as of 01/01/2025",
    'gestational_diabetes': "Whether the person has previously been diagnosed with gestational diabetes",
    # Family history of diabetes
    'family_history': "Family history of diabetes",
    'siblings': "Patient has siblings with diabetes",
    'children': "Patient has children with diabetes",
    # Medical history
    'hypertension': "Whether the patient has a history of treatment for hypertension",
    'hyperlipidaemia': "Whether the patient has a history of treatment for hyperlipidaemia",
    'ischemic_heart_disease': "Whether the patient has a history of treatment for ischemic heart disease",
    'cardiac_failure': "Whether the patient has a history of treatment for cardiac failure",
    'neuropathy': "Whether the patient has a history of treatment for neuropathy",
    'nephropathy': "Whether the patient has a history of treatment for nephropathy",
    'sympt_peripheral_neuropathy': "Whether the patient has a history of symptomatic peripheral neuropathy",
    'depression': "Whether the person has a history of depression",
    'stroke': "Whether the patient has had cerebral stroke due to vascular disease",
    'acute_myocardial_infarction': "Whether the patient has had an acute myocardial infarction",
    'transient_ischemic_attack': "Whether the patient has had transient ischemic attack",
    'cabg': "Whether the patient has had a coronary artery bypass graft",
    'cardiomyopathy': "Whether the patient has had ischaemic cardiomyopathy/cardiac failure",
    'autonomic_neuropathy': "Whether the patient has had autonomic neuropathy",
    # Problem list
    'retinopathy': "Whether the patient has retinopathy (either eye)",
    'lower_limb_problems': "Whether the patient has lower limb problems like Charcot foot, ulceration",
    'charcot_foot': "Whether the patient has Charcot foot (either foot)",
    'ulceration': "Whether the patient has a current foot ulcer on either foot",
    'claudication': "Whether the patient has experienced claudication",
    'nephropathy_indication': "Whether the patient has nephropathy (based on: eGFR, UACR indicators)",
    'cardiovascular_disease': "Whether the patient has cardiovascular disease (based on: MI, ischaemic cardiomyopathy/cardiac failure, CABG)",
    'cerebrovascular_disease': "Whether the patient has cerebrovascular disease (based on: AIT, stroke indicators)",
    # Medications
    'oral_contraceptive': "On oral contraceptives",
    'beta_blockers': "On beta blockers",
    'ace_inhibitor': "On ACE inhibitor",
    'calcium_channel_blocker': "On calcium channel blockers",
    'corticosteroids': "On corticosteroids",
    'thaizide': "On thiazide",
    'agr2_receptor_blocker': "On angiotensin 2 receptor blockers",
    'aspirin': "On aspirin treatment",
    'method_manage_t2dm': "The method used to manage patient's diabetes",
    # Bloods
    'sodium': "Patient's measured sodium level in blood (mmol/L)",
    'potassium': "Patient's measured potassium level in blood (mmol/L)",
    'chloride': "Patient's measured chloride level in blood (mmol/L)",
    'bicarbonate': "Patient's measured bicarbonate level in blood (mmol/L)",
    'creatinine': "Patient's measured serum creatinine level in blood (mmol/L)",
    'urea': "Patient's measured urea level in blood (mmol/L)",
    'haemoglobin': "Patient's measured haemoglobin level (g/L)",
    'albumin': "The patient's measured albumin level (g/L)",
    'white_cell_count': "Patient's measured white cell count (x10^9/L)",
    'platelets': "Patient's measured platelets count (x10^9/L)",
    'red_cell_count': "Patient's measured red blood cell count (x10^12/L)",
    'packed_cell_volume': "Patient's measured packed cell volume",
    'mean_cell_volume': "Patient's measured mean cell volume (fl)",
    'hba1c': "Patient's measured glycosylated haemoglobin (HbA1c) level as a percentage",
    'egfr': "Patient's albumin creatinine ratio in urine (mg/moll)",
    # Screening
    'islet_antibody': "Whether the person has had screening for Islet Antibody",
    'coeliac_antibody': "Whether the person has had screening for Coeliac Antibody",
    'thyroid_function': "Whether the person has had screening for Thyroid function",
    'thyroid_antibody': "Whether the person has had screening for Thyroid Antibody",
    'vitamin_b12': "Whether the person has had screening for Vitamin B12",
    'dka_diagnosis': "Whether the person has diagnosis of diabetic ketoacidosis ",
    # Admissions
    'admissions_in_2024': "Number of admissions to the hospital during 2024",
    'visits_before_2025': "Number of appointments before 01/01/2025",
    'outcome_admission_in_2025': "Whether the patient was hospitalised after 01/01/2025 - Yes/No",
    }

CLFS = {
    "Support Vector Machines": SVC(probability=True, kernel='linear', random_state=42),
    "Logistic Regression": LogisticRegression(solver='liblinear', max_iter=5000, random_state=42),
    "k Nearest Neighbours": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
}

CAT_FEATURES_ENCODED = {
    'RMH': 'hospital RMH',
    'WH': 'hospital WH',
    'F': 'sex F',
    'M': 'sex M',
    'CURRENT': 'smoker CURRENT',
    'FORMER': 'smoker FORMER',
    'NO': 'smoker NO',
    'UNKNOWN': 'smoker UNKNOWN',
    'BOTH INSULIN AND TABS': 'method BOTH INSULIN AND TABS',
    'DIET ONLY': 'method DIET ONLY',
    'INSULIN': 'method INSULIN',
    'INSULIN AND NON-INSULIN': 'method INSULIN AND NON-INSULIN',
    'NON-INSULIN': 'method NON-INSULIN',
    'OTHER': 'method OTHER',
    'TABLETS': 'method TABLETS',
    'UNKNOWN': 'method UNKNOWN'
}


def get_cols_for_display(df):
    dtype_display_names = {
        'category': 'Categorical',
        'float64': 'Continuous',
        'int64': 'Discrete',
        'bool': 'Boolean'
    }

    col_name = []
    col_type = []
    col_discription = []
    for col in df.columns:
        col_name.append(col)
        col_type.append(dtype_display_names[df[col].dtype.name])
        col_discription.append(COLUMN_DEFINITIONS[col])

    cols_for_display = pd.DataFrame(data={'column name': col_name, 'type': col_type , 'description': col_discription})
    cols_for_display.sort_values(by=['type', 'column name'], ignore_index=True, inplace=True)

    return cols_for_display

# Operator function lookup — used by apply_filter_mask
_OPS = {"==": eq, ">=": ge, ">": gt, "!=": ne, "<": lt, "<=": le}


def apply_filter_mask(df, col, op, val):
    """Return a boolean mask for df[col] <op> val.

    Uses operator functions instead of eval() for safety and clarity.
    op must be one of: '==', '>=', '>', '!=', '<', '<='
    """
    return _OPS[op](df[col], val)


def make_venn_diagram(subs1, subs2, inter, label1, label2, colors):
    """Display a two-set Venn diagram.

    subs1, subs2 : lists of subject keys for each condition
    inter        : list of keys in the intersection
    label1/2     : set label strings
    colors       : tuple of two matplotlib colours
    """
    plt.figure(figsize=(6, 6))
    plt.title("Venn diagram")
    venn2(
        subsets=(len(subs1), len(subs2), len(inter)),
        set_labels=(label1, label2),
        set_colors=colors,
    )
    plt.show()

def calculate_missing_values(df, report_zeros=False):
    # Calculate the number of missing values in each column and sort in descending order
    missing_values = df.isna().sum().sort_values(ascending=False)
    if not report_zeros:
        missing_values = missing_values[missing_values > 0]
    missing_values = missing_values.to_frame(name='Number of missing values')
    missing_values['Proportion of missing values'] = (missing_values['Number of missing values'] / len(df) * 100).round(2)
    return missing_values

def find_outliers(x):   
    percentile25 = x.quantile(0.25)
    percentile75 = x.quantile(0.75)
    
    iqr = percentile75-percentile25
    upper_limit = percentile75 + 1.5 * iqr
    lower_limit = percentile25 - 1.5 * iqr

    return x[(x <= lower_limit) | (x >= upper_limit)]

def get_categorical_features_encoded(preprocessor):
    fts = list(np.hstack(preprocessor.transformers_[0][1][1].categories_))
    return [CAT_FEATURES_ENCODED[ft] for ft in fts]

def name2model(model_name):
    return CLFS[model_name]

def checkbox2feature(checkboxes):
    features_list=[]
    for num_group in range(len(checkboxes)):
        for num_subgroup in range(len(checkboxes[num_group].children)):
            if checkboxes[num_group].children[num_subgroup].value:
                features_list.append(checkboxes[num_group].children[num_subgroup].description)
    return features_list

def score_cv(model, X, y):
    """Train and evaluate a model using cross-validation."""
    # Define scoring functions
    scoring = {
        'ROC AUC': 'roc_auc',
        'PR AUC': 'average_precision', 
        }

    # Define CV strategy
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    print("_" * 80)
    print("Training with %d-fold cross-validation:" % cv.n_splits)
    
    # Start timer
    start_time = time()
    
    scores = cross_validate(estimator=model, X=X, y=y, scoring=scoring, cv=cv, n_jobs=-1, return_train_score=False)
    
    train_time = time() - start_time
    
    # Print model name
    print(model)
        
    # Print training time    
    print("train time: %0.3fs" % train_time)

    # Print performance metrics
    for k,v in scores.items():
        if 'time' not in k:
            dataset, metric = k.split('_')
            print(f"Cross-validation {metric}: {v.mean():.3f} (+/- {v.std():.2f})")

def evaluate(model, X_dev, y_dev, X_holdout, y_holdout, threshold=0.5):
    model.fit(X_dev, y_dev)
    y_proba = model.predict_proba(X_holdout)[:, 1]

    # ROC and Precision-Recall curves in one figure
    auc_roc = roc_auc_score(y_holdout, y_proba)
    fpr, tpr, _ = roc_curve(y_holdout, y_proba)

    auc_pr = average_precision_score(y_holdout, y_proba)
    precision, recall, _ = precision_recall_curve(y_holdout, y_proba)

    _, axes = plt.subplots(1, 2, figsize=(10, 4))

    # ROC subplot
    axes[0].plot(fpr, tpr, color='orangered', linewidth=2, label=f'ROC AUC = {auc_roc:.3f}')
    axes[0].plot([0, 1], [0, 1], color='slategray', linestyle='--')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend()

    # Precision-Recall subplot
    axes[1].plot(recall, precision, color='forestgreen', linewidth=2, label=f'PR AUC = {auc_pr:.3f}')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend()

    plt.tight_layout()
    plt.show();

    y_pred = np.where(y_proba >= threshold, 1, 0)

    plt.figure(figsize=(10, 4))
    cmap = sns.light_palette('steelblue', as_cmap=True)
    class_names = ['Not admitted', 'Admitted']
    sns.heatmap(confusion_matrix(y_holdout, y_pred), 
                annot=True, fmt='d', annot_kws={'size': 16},
                cmap=cmap, cbar=False, 
                xticklabels=class_names, 
                yticklabels=class_names
                );

    plt.yticks(rotation=0)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion matrix");

    # print("Precision (positive predictive value): {:.0f}%".format(precision_score(y_holdout, y_pred)*100))
    # print("Recall (sensitivity): {:.0f}%".format(recall_score(y_holdout, y_pred)*100))

def select_features_cv(model, X, y, n_splits=10, tol=0.01, direction='forward'):
    # Define scoring functions
    scoring = 'roc_auc'

    # Define CV strategy
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    print("_" * 80)
    print("Training with %d-fold cross-validation:" % cv.n_splits)

    # Start timer
    start_time = time()

    sfs = SequentialFeatureSelector(
        estimator=model,
        # n_features_to_select=10,
        tol=tol,
        direction=direction,
        scoring=scoring,
        cv=cv,
        n_jobs=-1
    )
    sfs.fit(X, y)

    train_time = time() - start_time

    # Print training time    
    print("train time: %0.3fs" % train_time)

    # Print number of selected features
    print(f"Number of selected features: {sfs.get_support().sum()}")

    return sfs