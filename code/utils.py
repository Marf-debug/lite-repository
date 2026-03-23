import pandas as pd
from operator import eq, ge, gt, ne, lt, le
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

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

def find_outliers(x):   
    percentile25 = x.quantile(0.25)
    percentile75 = x.quantile(0.75)
    
    iqr = percentile75-percentile25
    upper_limit = percentile75 + 1.5 * iqr
    lower_limit = percentile25 - 1.5 * iqr

    return x[(x <= lower_limit) | (x >= upper_limit)]