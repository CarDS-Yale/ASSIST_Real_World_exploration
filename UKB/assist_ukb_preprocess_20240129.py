#%%
# Import selected packages
import pandas as pd
import numpy as np


#%%
# Split into chunks and search for selected procedures
chunksize = 10000  # size of chunks relies on your memory capacity
chunks = []

rename_csv = pd.read_csv("/path-to-dictionary/my_ukb_key.csv")
rename_dict = rename_csv.set_index('field.html')['col.name'].to_dict()

for chunk in pd.read_csv("/path-to-ukb-data/ukb47034.csv", chunksize=chunksize):

    chunk.rename(columns=rename_dict, inplace=True)
    cols_to_search = [col for col in chunk.columns if col.startswith('operative_procedures_opcs4_f41272')]
    mask = chunk[cols_to_search].isin(['U205', 'U194', 'U102', 'U106']).any(axis=1)
    filtered_chunk = chunk[mask]

    for code in ['U205', 'U194', 'U102', 'U106']:
        filtered_chunk[code] = filtered_chunk[cols_to_search].isin([code]).any(axis=1).astype(int)

    dates = {'U205_date': [], 'U194_date': [], 'U102_date': [], 'U106_date': []}

    for idx, row in filtered_chunk.iterrows():

        U205_date, U194_date, U102_date, U106_date = None, None, None, None

        for col in cols_to_search:

            suffix = col.split('_')[-1]
            
            if row[col] == 'U205' and U205_date is None:
                U205_date = chunk.loc[idx, f'date_of_first_operative_procedure_opcs4_f41282_0_{suffix}']
            elif row[col] == 'U194' and U194_date is None:
                U194_date = chunk.loc[idx, f'date_of_first_operative_procedure_opcs4_f41282_0_{suffix}']
            elif row[col] == 'U102' and U102_date is None:
                U102_date = chunk.loc[idx, f'date_of_first_operative_procedure_opcs4_f41282_0_{suffix}']
            elif row[col] == 'U106' and U106_date is None:
                U106_date = chunk.loc[idx, f'date_of_first_operative_procedure_opcs4_f41282_0_{suffix}']

        dates['U205_date'].append(U205_date)
        dates['U194_date'].append(U194_date)
        dates['U102_date'].append(U102_date)
        dates['U106_date'].append(U106_date)

    for code, date_list in dates.items():
        filtered_chunk[code] = date_list

    chunks.append(filtered_chunk)

#%%
# Concatenate all chunks into one DataFrame
sel = pd.concat(chunks, ignore_index=True)
sel


#%% Generate a subset for cardiac testing
test = sel[["U205", "U194", "U102", "U106", "U205_date", "U194_date", "U102_date", "U106_date", "eid"]]

# rename U205 = StressEcho, U194 = ETT, U102 = CCTA, U106 = MPI to respective names, same for date
test = test.rename(columns={"U205": "StressEcho", "U194": "ETT", "U102": "CCTA", "U106": "MPI"})
test = test.rename(columns={"U205_date": "StressEcho_date", "U194_date": "ETT_date", "U102_date": "CCTA_date", "U106_date": "MPI_date"})

# Identify the first test and respective date
for col in ["StressEcho_date", "ETT_date", "CCTA_date", "MPI_date"]:
    test[col] = pd.to_datetime(test[col])

date_cols = ["StressEcho_date", "ETT_date", "CCTA_date", "MPI_date"]
date_df = test[date_cols]
test["first_test_type"] = date_df.idxmin(axis=1)
test["first_test_date"] = date_df.min(axis=1)
test['first_test_type'] = test['first_test_type'].str.replace('_date', '')

# create flag for strategy (anatomical vs functional)  
test["strategy"] = np.where(test["first_test_type"].isin(["CCTA"]), "anatomical", "functional")

# summarize tests
test["first_test_type"].value_counts()


#%% Define the ids to be included in the ASSIST-UKB analysis
assist_ids = test["eid"].unique()
len(assist_ids)

#%% Load the ids to be excluded
excl_ids = pd.read_csv("/path-to-exclusion-file/ukb_exclude/w71033_2023-04-25.csv", header=None)
excl_ids = excl_ids[0].unique()
len(excl_ids)

#%% final assist_ids
assist_ids = set(assist_ids) - set(excl_ids)
len(assist_ids)


#%% Demographics
demo = sel[["eid", 
            "sex_f31_0_0",
            "ethnic_background_f21000_0_0",
            "body_mass_index_bmi_f21001_0_0",
            "date_of_attending_assessment_centre_f53_0_0", 
            "date_lost_to_followup_f191_0_0",
            "date_of_myocardial_infarction_f42000_0_0",
            "date_of_death_f40000_0_0",
            "underlying_primary_cause_of_death_icd10_f40001_0_0",
            "underlying_primary_cause_of_death_icd10_f40001_1_0",
            "year_of_birth_f34_0_0",
            "month_of_birth_f52_0_0"]]

demo["sex.2"] = np.where(sel["sex_f31_0_0"]==0, 1, 0) # originally female =0, thus reversing the coding
demo["bmi"] = demo["body_mass_index_bmi_f21001_0_0"]
demo["year_of_birth_f34_0_0"] = demo["year_of_birth_f34_0_0"].astype(int)
demo["month_of_birth_f52_0_0"] = demo["month_of_birth_f52_0_0"].astype(int)
demo["DOB"] = pd.to_datetime(demo["year_of_birth_f34_0_0"].astype(str) + "-" + demo["month_of_birth_f52_0_0"].astype(str))
demo["Date_attend"] = pd.to_datetime(demo["date_of_attending_assessment_centre_f53_0_0"])
demo["Date_lost_fu"] = pd.to_datetime(demo["date_lost_to_followup_f191_0_0"])
demo["Date_Death"] = pd.to_datetime(demo["date_of_death_f40000_0_0"])
demo["Date_MI"] = pd.to_datetime(demo["date_of_myocardial_infarction_f42000_0_0"])
demo["Date_Death_MI"] = demo[["Date_Death", "Date_MI"]].min(axis=1, skipna=True)

# code ethnicity
demo["ethnicity"] = np.where(demo["ethnic_background_f21000_0_0"].isin([1, 1001, 1002, 1003]), "White", np.where(
    demo["ethnic_background_f21000_0_0"].isin([2, 2001, 2002, 2003]), "Mixed", np.where(
        demo["ethnic_background_f21000_0_0"].isin([3, 5, 3001, 3002, 3003, 3004]), "Asian", np.where(
            demo["ethnic_background_f21000_0_0"].isin([4, 4001, 4002, 4003]), "Black", "Other"))))

# keep only eid, DOB, Death, MI, Death_MI, Date_lost_fu, Date_Death, Date_MI
demo = demo[["eid", "DOB", "ethnicity", "sex.2", "bmi", "Date_attend", "Date_lost_fu", "Date_Death_MI", "Date_Death", "Date_MI",
            "underlying_primary_cause_of_death_icd10_f40001_0_0", "underlying_primary_cause_of_death_icd10_f40001_1_0"]]
demo


#%% Merge test and demo
merged = pd.merge(test, demo, on="eid", how="left")
merged

merged["MI"] = np.where((merged["Date_MI"].notnull()) & (merged["Date_MI"] > merged["first_test_date"]), 1, 0)
merged["Death"] = np.where((merged["Date_Death"].notnull()) & (merged["Date_Death"] > merged["first_test_date"]), 1, 0)
merged["Death_MI"] = np.where((merged["Date_Death_MI"].notnull()) & (merged["Date_Death_MI"] > merged["first_test_date"]), 1, 0)

# create time to event; if event = 1, then take the time diff, otherwise, take "2020-12-31", same for death
merged["Time2MI"] = np.where(merged["MI"] == 1, (merged["Date_MI"] - merged["first_test_date"]).dt.days, 
                             (pd.to_datetime("2020-12-31") - merged["first_test_date"]).dt.days)
merged["Time2Death"] = np.where(merged["Death"] == 1, (merged["Date_Death"] - merged["first_test_date"]).dt.days,
                                (pd.to_datetime("2020-12-31") - merged["first_test_date"]).dt.days)
merged["Time2Death_MI"] = np.where(merged["Death_MI"] == 1, (merged["Date_Death_MI"] - merged["first_test_date"]).dt.days,
                                      (pd.to_datetime("2020-12-31") - merged["first_test_date"]).dt.days)

merged["age"] = ((merged["first_test_date"] - merged["DOB"]).dt.days)/365.25

# Define CV death
merged["CV_death"] = np.where((merged["underlying_primary_cause_of_death_icd10_f40001_0_0"].str.startswith("I")) | 
                            (merged["underlying_primary_cause_of_death_icd10_f40001_1_0"].str.startswith("I")), 1, 0)

# Define CVDeath_MI
merged["CVDeath_MI"] = np.where((merged["CV_death"] == 1) | (merged["MI"] == 1), 1, 0)

                                   

# %%
# Define variables of interest
# Smoking
smoking = sel[["eid", "smoking_status_f20116_0_0"]]
smoking["smoking_status_f20116_0_0"] = smoking["smoking_status_f20116_0_0"].astype("category")
smoking["smoking_status_f20116_0_0"] = smoking["smoking_status_f20116_0_0"].cat.rename_categories({-3: "prefer not to answer", 0: "never", 1: "previous", 2: "current"})
# create flag for former smokers; smoke.3 = 1 if previous smoker, 0 otherwise
smoking["smoke.3"] = np.where(smoking["smoking_status_f20116_0_0"] == "previous", 1, 0)
# create flag for current smokers; smoke.2 = 1 if current smoker, 0 otherwise
smoking["smoke.2"] = np.where(smoking["smoking_status_f20116_0_0"] == "current", 1, 0)
# keep only eid, smoke.2, smoke.3
smoking = smoking[["eid", "smoke.2", "smoke.3"]]



#%% Extract comorbidities
# first add test_date
sel = sel.merge(test, on="eid", how="left")

htn_list = ['I10', 'I11', 'I110', 'I119', 'I12', 'I120', 'I129', 'I13', 'I130', 'I131',
            'I132', 'I139', 'I674', 'O10', 'O100', 'O101', 'O102', 'O103', 'O109', 'O11',
            
            '401', '4010', '4011', '4019', '402', '4020', '4021', '4029', '403', '4030',
            '4031', '4039', '404', '4040', '4041', '4049', '6420', '6422', '6427', '6429']

dm_list = ['E10', 'E100', 'E101', 'E102', 'E103', 'E104', 'E105', 'E106', 'E107', 'E108',
           'E109', 'E11', 'E110', 'E111', 'E112', 'E113', 'E114', 'E115', 'E116', 'E117',
           'E118', 'E119', 'E12', 'E120', 'E121', 'E122', 'E123', 'E124', 'E125', 'E126',
           'E127', 'E128', 'E129', 'E13', 'E130', 'E131', 'E132', 'E133', 'E134', 'E135',
           'E136', 'E137', 'E138', 'E139', 'E14', 'E140', 'E141', 'E142', 'E143', 'E144',
           'E145', 'E146', 'E147', 'E148', 'E149', 'O240', 'O241', 'O242', 'O243', 'O249',
    
           '250', '2500', '25000', '25001', '25009', '2501', '25010', '25011', '25019', '2502',
           '25020', '25021', '25029', '2503', '2504', '2505', '2506', '2507', '2509', '25090',
           '25091', '25099', '6480']

ckd_list = ['I12', 'I120', 'I13', 'I130', 'I131', 'I132', 'I139', 'N18', 'N180', 'N181',
            'N182', 'N183', 'N184', 'N185', 'N188', 'N189', 'Z49', 'Z490', 'Z491', 'Z492',
            '403', '4030', '4031', '4039', '404', '4040', '4041', '4049', '585', '5859',
            '6421', '6462']

heart_failure_list = ['I110', 'I130', 'I132', 'I50', 'I500', 'I501', 'I509',
                     
                      '428','4280','4281','4289']

ami_list = ['I21', # Acute myocardial infarction
            'I22', # Subsequent myocardial infarction
            'I23', # Certain current complications following acute myocardial infarction
            'I240', # Acute coronary thrombosis not resulting in myocardial infarction
            'I248', # Other forms of acute ischemic heart disease
            'I249' # Acute ischemic heart disease, unspecified
            '410', '4110', '4111', '4118']

ihd_list = ['I20', 'I200', 'I208', 'I209', 'I21', 'I210', 'I211', 'I212', 'I213',
            'I214', 'I219', 'I21X', 'I22', 'I220', 'I221', 'I228', 'I229', 'I23', 'I230',
            'I231', 'I232', 'I233', 'I234', 'I235', 'I236', 'I238', 'I24', 'I240', 'I241',
            'I248', 'I249', 'I25', 'I250', 'I251', 'I252', 'I255', 'I256', 'I258', 'I259',
            'Z951', 'Z955',
            '410', '4109', '411', '4119', '412', '4129', '413', '4139', '414', '4140',
            '4148', '4149']

pad_list = ['I702', 'I7020', 'I7021', 'I742', 'I743', 'I744'
            '4402', '4442']

stroke_list=['G45', 'G450', 'G451', 'G452', 'G453', 'G454', 'G458', 'G459', 'I63', 
             'I630', 'I631', 'I632', 'I633', 'I634', 'I635', 'I638', 'I639', 'I64',
             'I65', 'I650', 'I651', 'I652', 'I653', 'I658', 'I659', 'I66', 'I660',
             'I661', 'I662', 'I663', 'I664', 'I668', 'I669', 'I672', 'I693', 'I694',
             '433', '4330', '4331', '4332', '4333', '4338', '4339', '434', '4340',
             '4341', '4349', '435', '4359', '437', '4370', '4371']

# Create a list of columns to search for each diagnosis
icd9_cols = [col for col in sel.columns if col.startswith('diagnoses_icd9_f41271_0_')]
icd10_cols = [col for col in sel.columns if col.startswith('diagnoses_icd10_f41270_0_')]

# Generate a list of date columns that correspond to each icd9 and icd10 column
date_icd9_cols = [col.replace('diagnoses_icd9_f41271_0_', 'date_of_first_inpatient_diagnosis_icd9_f41281_0_') for col in icd9_cols]
date_icd10_cols = [col.replace('diagnoses_icd10_f41270_0_', 'date_of_first_inpatient_diagnosis_icd10_f41280_0_') for col in icd10_cols]

def check_diagnosis(row, icd_list, date_prefix, test_date_col='first_test_date'):
    for col in icd9_cols + icd10_cols:
        # If a diagnosis is matched
        if row[col] in icd_list:
            # Extract the index from the column name
            idx_suffix = '_' + col.split('_')[-2] + '_' + col.split('_')[-1]

            if 'icd9' in col:
                date_col = 'date_of_first_inpatient_diagnosis_icd9_f41281' + idx_suffix
            else:
                date_col = 'date_of_first_inpatient_diagnosis_icd10_f41280' + idx_suffix

            # Convert date column to Timestamp if it's not null and not already a Timestamp
            if pd.notnull(row[date_col]) and not isinstance(row[date_col], pd.Timestamp):
                row[date_col] = pd.to_datetime(row[date_col])

            # If the diagnosis date is missing or before the first_test_date, it's a valid diagnosis
            if pd.isnull(row[date_col]) or row[date_col] < row[test_date_col]:
                return (1, row[date_col])
    return (0, None)

# Apply the function and create new columns for both diagnosis and its date
conditions = [('htn', htn_list), ('diab', dm_list), ('ckd', ckd_list), ('hf', heart_failure_list), 
              ('ami', ami_list), ('ihd', ihd_list), ('pad', pad_list), ('stroke', stroke_list)]

for condition, icd_list in conditions:
    sel[f'{condition}.1'], sel[f'{condition}_firstdate'] = zip(*sel.apply(lambda row: check_diagnosis(row, icd_list, 'f41280_0'), axis=1))


#%% Pull PCI/CABG history
cabg_list = ['3043', 'K40', 'K41', 'K42', 'K43', 'K44', 'K45', 'K46']
pci_list = ['K49', 'K50', 'K75', 'K78']

# Columns to search
opcs4_cols = [col for col in sel.columns if col.startswith('operative_procedures_opcs4_f41272_0_')]
opcs3_cols = [col for col in sel.columns if col.startswith('operative_procedures_opcs3_f41273_0_')]
cols_to_search = opcs4_cols + opcs3_cols

def get_date(row, item_list):
    for col in cols_to_search:
        # If a procedure is matched
        if any(item in str(row[col]) for item in item_list):
            idx_suffix = col.split('_')[-1]  # Extract the index suffix without _0_ prefix
            
            # Match the date column based on the procedure
            if 'opcs4' in col:
                date_col = 'date_of_first_operative_procedure_opcs4_f41282_0_' + idx_suffix
            else:
                date_col = 'date_of_first_operative_procedure_opcs3_f41283_0_' + idx_suffix
            
            # Ensure both dates are in the Timestamp format
            procedure_date = pd.to_datetime(row[date_col], errors='coerce')
            first_test_date = pd.to_datetime(row['first_test_date'], errors='coerce')
            
            # Check if the column exists and the procedure date is before the first_test_Date
            if date_col in row.index and not pd.isna(procedure_date) and procedure_date < first_test_date:
                return procedure_date
    return np.nan

# Get the dates for CABG and PCI
sel['cabg_firstdate'] = sel.apply(lambda row: get_date(row, cabg_list), axis=1)
sel['pci_firstdate'] = sel.apply(lambda row: get_date(row, pci_list), axis=1)

# Check for PCI and CABG procedures
def check_partial_matches(row, item_list):
    return any(isinstance(cell, str) and any(item in cell for item in item_list) for cell in row)

sel['pci'] = sel[cols_to_search].apply(lambda row: check_partial_matches(row, pci_list), axis=1).astype(int)
sel['cabg'] = sel[cols_to_search].apply(lambda row: check_partial_matches(row, cabg_list), axis=1).astype(int)

# Update ihd.1 column and create a combined cabg_pci column
sel["ihd.1"] = np.where((sel["cabg"]==1)|(sel["pci"]==1), 1, sel["ihd.1"])
sel["cabg_pci"] = np.where((sel["cabg"]==1)|(sel["pci"]==1), 1, 0)

# Pull the comorbidities of interest
comorbidities = sel[["eid", "htn.1", "htn_firstdate", "diab.1", "diab_firstdate", "ckd.1", "ckd_firstdate",
                     "hf.1", "hf_firstdate", "ami.1", "ami_firstdate", "ihd.1", "ihd_firstdate", "pad.1",
                     "pad_firstdate", "stroke.1", "stroke_firstdate",
                     "pci", "cabg", "cabg_pci", "cabg_firstdate", "pci_firstdate"]]

#%% Pull medications

# Medications = 20003; treatmentmedication_code_f20003_0_0
beta_blocker = ["metoprolol", "betaloc", "lopressor", "lopresor",
                "atenolol", "tenormin",
                "propranolol", "angilol", "inderal",
                "carvedilol", 
                "bisoprolol", "cardicor", "emcor",
                "labetalol", "trandate",
                "pindolol", "nebivolol",
                "penbutolol", "acebutolol", "apsolol", "berkolol", "betaxolol", "celiprolol",
                "carteolol", "levobunolol", "oxprenolol", "penbutolol", "prindolol", "sloprolol",
                "nadolol",
                "sotalol"]

statin = ["atorvastatin", "lipitor", "simvastatin", "zocor", "pravastatin", "pravachol",
            "rosuvastatin", "crestor", "fluvastatin", "lescol", "lovastatin", "mevacor",
            "pitavastatin", "livalo", "statin"]

antiplatelet = ["aspirin", "clopidogrel", "prasugrel", "ticagrelor", "ticlopidine", "cilostazol", "dipyridamole", "abciximab", "eptifibatide", "tirofiban"]

# read .csv with medication names
medication_names = pd.read_csv("/home/eo287/assist/dictionary/coding4.tsv", delimiter="\t")

# replace list with codes
bb_list = medication_names[medication_names['meaning'].str.contains('|'.join(beta_blocker), case=False)]['coding'].tolist()
statin_list = medication_names[medication_names['meaning'].str.contains('|'.join(statin), case=False)]['coding'].tolist()
antiplatelet_list = medication_names[medication_names['meaning'].str.contains('|'.join(antiplatelet), case=False)]['coding'].tolist()

# Create a list of columns to search for each treatment
treatment_cols = [col for col in sel.columns if col.startswith('treatmentmedication_code_f20003_')]

# Create a list of columns to search for each treatment
treatment_cols = [col for col in sel.columns if col.startswith('treatmentmedication_code_f20003_')]

# Create a function to apply
def check_contains(row, check_list):
    return row.astype(str).apply(lambda x: any(str(item) in x for item in check_list))

# Create 'beta_blocker' column, check if any value in beta_blocker list is in any of the columns to search
sel['betablk_1.1'] = sel[treatment_cols].apply(lambda row: check_contains(row, bb_list), axis=1).any(axis=1).astype(int)

# Create 'statin' column, check if any value in statin list is in any of the columns to search
sel['statin_1.1'] = sel[treatment_cols].apply(lambda row: check_contains(row, statin_list), axis=1).any(axis=1).astype(int)

# Create 'antiplatelet' column, check if any value in antiplatelet list is in any of the columns to search
sel['antiplatelet_1.1'] = sel[treatment_cols].apply(lambda row: check_contains(row, antiplatelet_list), axis=1).any(axis=1).astype(int)

# Pull the medications of interest
medications = sel[["eid", "betablk_1.1", "statin_1.1", "antiplatelet_1.1"]]

#%% Pull lab values
labs = sel[["eid", "cholesterol_f30690_0_0", 
            "cholesterol_assay_date_f30691_0_0",
            "hdl_cholesterol_f30760_0_0", 
            "hdl_cholesterol_assay_date_f30761_0_0"]]
# transform chol from mmol/L to mg/dL
labs["choles"] = labs["cholesterol_f30690_0_0"] * 38.67
labs["hdl"] = labs["hdl_cholesterol_f30760_0_0"] * 38.67

# emrge with test date
labs = pd.merge(labs, test, on="eid", how="left")

# if date of chol more than 1 month after date of test, then switch to nan, same for hdl
from pandas.tseries.offsets import DateOffset
labs["chol"] = np.where(labs["cholesterol_assay_date_f30691_0_0"] > (labs["first_test_date"] + DateOffset(months=12)), np.nan, labs["chol"])
labs["hdl"] = np.where(labs["hdl_cholesterol_assay_date_f30761_0_0"] > (labs["first_test_date"] + DateOffset(months=12)), np.nan, labs["hdl"])

# keep only eid, chol, hdl
labs = labs[["eid", "choles", "hdl"]]

# summarize missing
labs.isnull().sum()


#%% Merge all data
# Merge all data
merged = pd.merge(merged, comorbidities, on="eid", how="left")
merged = pd.merge(merged, medications, on="eid", how="left")
merged = pd.merge(merged, smoking, on="eid", how="left")
merged = pd.merge(merged, labs, on="eid", how="left")
merged = merged[merged["eid"].isin(assist_ids)]
len(merged)
merged.to_csv("/path-to-ukb-file/ukb_v2_20240115.csv", index=False)

