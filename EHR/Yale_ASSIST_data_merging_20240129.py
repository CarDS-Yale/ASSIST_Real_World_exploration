# %% [markdown]
# # Load the processed files

# %%
import pandas as pd
import numpy as np

# %%
# Load files
imaging = pd.read_csv('/ehr-extract/ASSIST_first_imaging_EHR_20240116.csv')
censoring = pd.read_csv("/ehr-extract/ASSIST_censoring.csv")
comorbidities = pd.read_csv("/ehr-extract/ASSIST_comorbidities_dates_20231102.csv")
pci_cabg = pd.read_csv("/ehr-extract/ASSIST_cpt_pci_cabg_history.csv")
bp = pd.read_csv("/ehr-extract/bp_sel.csv")
bmi = pd.read_csv("/ehr-extract/ASSIST_bmi_joined.csv")
labs = pd.read_csv("/ehr-extract/ASSIST_labs.csv")
meds = pd.read_csv("/ehr-extract/ASSIST_meds_with_antihypertensives_and_new_meds.csv")
smoking = pd.read_csv("/ehr-extract/ASSIST_smoking.csv")
outcomes = pd.read_csv("/ehr-extract/ASSIST_outcome_dates.csv")

# %%
# from each one of the above files, drop duplicate rows for pat_mrn_id
imaging.drop_duplicates(subset='PAT_MRN_ID', keep='first', inplace=True)
censoring.drop_duplicates(subset='PAT_MRN_ID', keep='first', inplace=True)
comorbidities.drop_duplicates(subset='PAT_MRN_ID', keep='first', inplace=True)
pci_cabg.drop_duplicates(subset='PAT_MRN_ID', keep='first', inplace=True)
bp.drop_duplicates(subset='PAT_MRN_ID', keep='first', inplace=True)
bmi.drop_duplicates(subset='PAT_MRN_ID', keep='first', inplace=True)
labs.drop_duplicates(subset='PAT_MRN_ID', keep='first', inplace=True)
meds.drop_duplicates(subset='PAT_MRN_ID', keep='first', inplace=True)
smoking.drop_duplicates(subset='PAT_MRN_ID', keep='first', inplace=True)
outcomes.drop_duplicates(subset='PAT_MRN_ID', keep='first', inplace=True)

# %%
# Merge all files on "PAT_MRN_ID" - do left join starting with imaging
df = imaging.merge(censoring, on="PAT_MRN_ID", how="left")
df = df.merge(comorbidities, on="PAT_MRN_ID", how="left")
df = df.merge(pci_cabg, on="PAT_MRN_ID", how="left")
df = df.merge(bp, on="PAT_MRN_ID", how="left")
df = df.merge(bmi, on="PAT_MRN_ID", how="left")
df = df.merge(labs, on="PAT_MRN_ID", how="left")
df = df.merge(meds, on="PAT_MRN_ID", how="left")
df = df.merge(smoking, on="PAT_MRN_ID", how="left")
df = df.merge(outcomes, on="PAT_MRN_ID", how="left")

# make all lower case
df.columns = df.columns.str.lower()

# %%
# drop duplicate rows for "pat_mrn_id"
df = df.drop_duplicates(subset="pat_mrn_id", keep="first")
len(df)

# %%
df["cardiac_test_grp"].value_counts()

# %%
# Defining a dictionary to map the race values
# Define Race
df["race"] = np.where(df["patient_race_all"].str.contains("White|Caucasian|Middle Eastern or Northern African|Middle Eastern or Northern African", case=False), "White", np.where(
                        df["patient_race_all"].str.contains("Black|African American", case=False), "Black", np.where(
                            df["patient_race_all"].str.contains("Asian|Pacific Islander|Chinese|Japanese|Vietnamese|Filiino|Samoan|Korean|Guamanian", case=False), "Asian", np.where(
                                df["patient_race_all"].str.contains("Unknown"), "Unknown", "Other"))))
print(df["race"].value_counts())

# Define Ethnicity
df["ethnicity"] = np.where(df["patient_ethnicity"].isin(["Hispanic or Latino", "Hispanic or Latino - Cuban", "Cuban", "Mexican, Mexican American, Chicano/a", "Puerto Rican", "Hispanic or Latina/o/x",
                                                         "Hispanic or Latino - Mexican", "Hispanic or Latino - Puerto Rican", 
                                                         "Hispanic or Latino - Other", "Hispanic or Latino - Unknown"]), "Hispanic or Latina/o/x", np.where(df["patient_ethnicity"]=="Not Hispanic or Latina/o/x", "Non-Hispanic", "Unknown"))

print(df["ethnicity"].value_counts())

# Create prior IHD flag
df["prior_ihd"] = np.where(df["ihd_dx_date_first"].notnull(), 1, 0)
print("In total, there are", df["prior_ihd"].sum(), "patients with prior IHD")

# define strategy
df["strategy"] = np.where(df["cardiac_test_grp"].isin(["nuc", "echo_stress", "cmr_stress", "ett"]), "functional", "anatomical")
print(df["strategy"].value_counts())

# %%
# if smoker_active, smoker_former, htn, ami, ihd, stroke, heart_failure, dm, ckd, pad, stroke are missing then replace with zero
df["smoker_active"] = df["smoker_active"].fillna(0)
df["smoker_former"] = df["smoker_former"].fillna(0)
df["htn"] = df["htn"].fillna(0)
df["ami"] = df["ami"].fillna(0)
df["ihd"] = df["ihd"].fillna(0)
df["stroke"] = df["stroke"].fillna(0)
df["heart_failure"] = df["heart_failure"].fillna(0)
df["dm"] = df["dm"].fillna(0)
df["ckd"] = df["ckd"].fillna(0)
df["pad"] = df["pad"].fillna(0)

df["antihypertensive"] = df["antihypertensive"].fillna(0)
df["statin"] = df["statin"].fillna(0)
df["antiplatelet"] = df["antiplatelet"].fillna(0)
df["beta_blocker"] = df["beta_blocker"].fillna(0)

df["antihypertensive_within1y"] = df["antihypertensive_within1y"].fillna(0)
df["statin_within1y"] = df["statin_within1y"].fillna(0)
df["antiplatelet_within1y"] = df["antiplatelet_within1y"].fillna(0)
df["beta_blocker_within1y"] = df["beta_blocker_within1y"].fillna(0)

# %%
# cabg, pci - combine definitions; also add ihd to those with pci/cabg flag
df["cabg"] = np.where(df["cabg_icd"]==1, 1, df["cabg"])
df["pci"] = np.where(df["pci_icd"]==1, 1, df["pci"])
df["ihd"] = np.where((df["pci_icd"]==1)|(df["cabg_icd"]==1), 1, df["ihd"])
df = df.drop(["cabg_icd", "pci_icd"], axis=1)
df["cabg_pci"] = np.where((df["cabg"]==1)|(df["pci"]==1), 1, 0)
df["cabg_pci"].value_counts()

# %%
# Extract float from string
df["time_to_censor"] = df["time_to_censor"].str.extract(r'(\d+)')
df["time_to_censor"] = df["time_to_censor"].astype(float)

# %%
# Exclude if death_date < birth_date
df = df[~(df["death_date"] <= df["birth_date"])]
df = df[~(df["death_date"] <= df["test_date_x"])]
df = df[~(df["fu_ami_dx_date_first"] <= df["test_date_x"])]
df = df[~(df["fu_stroke_dx_date_first"] <= df["test_date_x"])]
df = df[~(df["fu_heart_failure_dx_date_first"] <= df["test_date_x"])]

# %%
# Define outcomes; if missing date, then replace with 2023-08-03
df["fu_heart_failure_dx_date_first"] = df["fu_heart_failure_dx_date_first"].fillna(df["censor_date"])
df["fu_ami_dx_date_first"] = pd.to_datetime(df["fu_ami_dx_date_first"]).fillna(df["censor_date"])
df["fu_stroke_dx_date_first"] = pd.to_datetime(df["fu_stroke_dx_date_first"]).fillna(df["censor_date"])

# Calculate time to event for each outcome
df["time_to_heart_failure"] = (pd.to_datetime(df["fu_heart_failure_dx_date_first"]) - pd.to_datetime(df["test_date_x"])).dt.days
df["time_to_ami"] = (pd.to_datetime(df["fu_ami_dx_date_first"]) - pd.to_datetime(df["test_date_x"])).dt.days
df["time_to_stroke"] = (pd.to_datetime(df["fu_stroke_dx_date_first"]) - pd.to_datetime(df["test_date_x"])).dt.days

# Replace missing values with 0
df["fu_heart_failure"] = df["fu_heart_failure"].fillna(0)
df["fu_ami"] = df["fu_ami"].fillna(0)
df["fu_stroke"] = df["fu_stroke"].fillna(0)

# Please summarize the number of patients with each outcome
print("Number of patients with heart failure: ", len(df[df["fu_heart_failure"] == 1]))
print("Number of patients with AMI: ", len(df[df["fu_ami"] == 1]))
print("Number of patients with stroke: ", len(df[df["fu_stroke"] == 1]))

# %%
# Create composite
df["MACE2"] = np.where((df["death"] == 1) | (df["fu_ami"] == 1), 1, 0)
df["MACE3"] = np.where((df["death"] == 1) | (df["fu_ami"] == 1) | (df["fu_stroke"] == 1), 1, 0)
df["MACE4"] = np.where((df["death"] == 1) | (df["fu_ami"] == 1) | (df["fu_stroke"] == 1) | (df["fu_heart_failure"] == 1), 1, 0)

# Create time to MACE - if 1, take the minimum of time_to_censor, time_to_heart_failure, time_to_ami, time_to_stroke, else take time_to_censor
df["time_to_MACE2"] = np.where(df["MACE2"] == 1, df[["time_to_censor", "time_to_ami"]].min(axis=1), df["time_to_censor"])
df["time_to_MACE3"] = np.where(df["MACE3"] == 1, df[["time_to_censor", "time_to_ami", "time_to_stroke"]].min(axis=1), df["time_to_censor"])
df["time_to_MACE4"] = np.where(df["MACE4"] == 1, df[["time_to_censor", "time_to_heart_failure", "time_to_ami", "time_to_stroke"]].min(axis=1), df["time_to_censor"])

# Summarize
print("Number of patients with MACE2: ", len(df[df["MACE2"] == 1]))
print("Number of patients with MACE3: ", len(df[df["MACE3"] == 1]))
print("Number of patients with MACE4: ", len(df[df["MACE4"] == 1]))

# %%
# If BMI > 100, then replace with missing
df["bmi"] = np.where(df["bmi"] > 100, np.nan, df["bmi"])

# %%
# remove if test_date_x is missing
df = df[~df["test_date_x"].isnull()]

# %%
# show all columns
pd.set_option('display.max_columns', None)

# %%
# drop duplicates from df (for pat_mrn_id)
df_final = df.drop_duplicates(subset=['pat_mrn_id'], keep='first')

# %%
# exclude if age < 18
df_final = df_final[~(df_final["age_test"] < 18)]

# %%
# Define dataset for analysis
df_sel = df_final[["pat_mrn_id", "accession_num", "order_narrative", "prior_ihd", "cardiac_test", "cardiac_test_spec", "strategy", "age_test", "sex", "race", "ethnicity", 
             "attenuation_correction", "pharmacological", "year", "inpatient", 'sbp', 'dbp', 'abnormal', 'abnormal_or_positive', 
             "htn", "ami", "ihd", "heart_failure", "dm", "ckd", "pad", "stroke", "cabg", "pci", "cabg_pci", "bmi", "cholesterol", "hdl", 
             "statin", "antiplatelet", "beta_blocker", 'antihypertensive', 
             "statin_within1y", "antiplatelet_within1y", "beta_blocker_within1y", 'antihypertensive_within1y',
             "smoker_active", "smoker_former",
             "death", "time_to_censor", "fu_heart_failure", "fu_ami", "fu_stroke", "MACE2", "MACE3", "MACE4", "time_to_heart_failure", "time_to_ami", "time_to_stroke", "time_to_MACE2", "time_to_MACE3", "time_to_MACE4"]]

# Write df_sel to csv
df_sel.to_csv("/ehr-extract/ASSIST_Yale_final_20240116.csv", index=False)


