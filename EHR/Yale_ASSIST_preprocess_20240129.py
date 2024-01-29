# %% [markdown]
# ## Defining the cohort

# %%
# Load the required packages
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# %%
# Import data from the EHR extract
imaging = pd.read_csv('/ehr-extract/2457789_CarDS_ASSIST_Imaging.txt', delimiter='\t', on_bad_lines='skip')

# %%
# Load all values and manually inspect them to ensure complete capture of different tests and no missing data
pd.set_option('display.max_rows', None)
#imaging[imaging['PROC_NAME'].str.contains('stress|coronary|perfusion|cardiac|heart|treadmill', case=False)]["PROC_NAME"].value_counts()
all_tests_names = imaging[imaging['PROC_NAME'].str.contains('stress|coronary|perfusion|cardiac|heart|treadmill', case=False)]["PROC_NAME"].unique()

# %%
# Define the testing modality
nuc = ["NM MYOCARDIAL PERFUSION SPECT (STRESS AND REST) WITH REGADENOSON",
       "NM MYOCARDIAL PERFUSION SPECT (STRESS AND REST) WITH EXERCISE",
       "PET CT MYOCARDIAL PERFUSION (STRESS AND REST) W REGADENOSON (YH)"
       #... add extra types here
       ]

spect = ["NM MYOCARDIAL PERFUSION SPECT (STRESS AND REST) WITH REGADENOSON",
       "NM MYOCARDIAL PERFUSION SPECT (STRESS AND REST) WITH EXERCISE",
       "NM CARDIAC STRESS TEST - CARDIOLOGIST RESULT (YH)",
       "NM MYOCARDIAL PERFUSION SPECT (STRESS ONLY) WITH EXERCISE",
       "NM MYOCARDIAL PERFUSION SPECT (STRESS ONLY) WITH REGADENOSON"
       #... add extra types here
       ]

pet = ["PET CT MYOCARDIAL PERFUSION (STRESS AND REST) W REGADENOSON (YH)",
       "PET CT MYOCARDIAL PERF (STRESS REST) WITH ADENOSINE (YH)",
       "PET CT MYOCARDIAL PERFUSION (STRESS AND REST) W DOUBUTAMINE (YH)"
       #... add extra types here
       ]

ett = ["EXERCISE STRESS TEST, NO IMAGING"]   

echo_stress = ["STRESS ECHO WITH EXERCISE W COMP DOPP AND CFI IF IMAGE ENHANCE", 
              "STRESS ECHO WITH EXERCISE IF INDICATED IMAGE ENHANCEMENT",
              "STRESS ECHO WITH EXERCISE W COMP DOPP IF IND IMAGE ENHANCE",
              "STRESS ECHO WITH DOBUTAMINE IF INDICATED IMAGE ENHANCEMENT"
              #... add extra types here
              ]

angio = ["CORONARY ANGIOGRAGHY",
         #... add extra types here e.g. LEFT HEART CATH, CARDIAC CATHETERIZATION
]

cmr_stress = ["MRI HEART W STRESS W  WO IV CONTRAST (BH YH GH)"
              #... add extra types here
              ]

ccta = ["CTA CORONARY", 
        #... add extra types here E.G. OSF CT HEART/CARDIAC, CTA cardiac if appropriate
       ]

# %%
# Define the stress type

pharm = ["NM MYOCARDIAL PERFUSION SPECT (STRESS AND REST) WITH REGADENOSON",
       "NM MYOCARDIAL PERFUSION SPECT (STRESS ONLY) WITH REGADENOSON",
       "NM MYOCARDIAL PERFUSION SPECT (STRESS AND REST) WITH ADENOSINE",
       "NM MYOCARDIAL PERFUSION SPECT (STRESS AND REST) WITH DOBUTAMINE",
       #... add extra types here

       "MRI HEART W STRESS W  WO IV CONTRAST (BH YH GH)",
       #... add extra types here

       "STRESS ECHO WITH DOBUTAMINE W COMP DOPP CFI IF IND IMG ENHANCE",
       #... add extra types here
       ]

exer = ["NM MYOCARDIAL PERFUSION SPECT (STRESS AND REST) WITH EXERCISE",
        #... add extra types here

       "STRESS ECHO WITH EXERCISE W COMP DOPP AND CFI IF IMAGE ENHANCE", 
       #... add extra types here

       "EXERCISE STRESS TEST, NO IMAGING"
       #... add extra types here
       ]

# %%
# Extract data for attenuation correction (based on the presence of an associated CT read)
imaging[imaging["PROC_NAME"].str.contains("CT READ")]["PROC_NAME"].value_counts()
imaging["attenuation_correction"] = np.where(((imaging["PROC_NAME"]=="NM SPECT CT READ")|(imaging["PROC_NAME"]=="PET/CT STRESS CT READ")), 1, np.nan)
atten = imaging[imaging["attenuation_correction"]==1]
atten_sel = atten[["PAT_MRN_ID", "END_EXAM_DTTM"]]

# %%
# Define type of cardiac test
imaging["cardiac_test"] = np.where(imaging["PROC_NAME"].isin(spect), "spect", np.where(imaging["PROC_NAME"].isin(pet), "pet",
                                    np.where(imaging["PROC_NAME"].isin(ccta), "ccta",
                                                np.where(imaging["PROC_NAME"].isin(angio), "angio",
                                                            np.where(imaging["PROC_NAME"].isin(ett), "ett",
                                                                        np.where(imaging["PROC_NAME"].isin(echo_stress), "echo_stress",
                                                                                    np.where(imaging["PROC_NAME"].isin(cmr_stress), "cmr_stress", "other")))))))
imaging = imaging[imaging['cardiac_test']!="other"]
imaging["cardiac_test"].value_counts()

# %%
# Extract nuclear + atten_correction
nuclear = imaging[(imaging["cardiac_test"]=="pet")| (imaging["cardiac_test"]=="spect")]
atten_merged = nuclear.merge(atten_sel, on="PAT_MRN_ID", how="left", suffixes=('_nuclear', '_atten'))

atten_merged['END_EXAM_DTTM_nuclear'] = pd.to_datetime(atten_merged['END_EXAM_DTTM_nuclear'])
atten_merged['END_EXAM_DTTM_atten'] = pd.to_datetime(atten_merged['END_EXAM_DTTM_atten'])
atten_merged['atten_done'] = ((atten_merged['END_EXAM_DTTM_atten'] - atten_merged['END_EXAM_DTTM_nuclear']).abs() <= pd.Timedelta('1D')).astype(int)
atten_merged['atten_done'].value_counts()

idx_to_keep = atten_merged.groupby('ACCESSION_NUM')['atten_done'].idxmax()
atten_merged_unique = atten_merged.loc[idx_to_keep]
atten_merged_unique['atten_done'].value_counts()

atten_acc_nums = atten_merged_unique[atten_merged_unique['atten_done']==1]["ACCESSION_NUM"].unique()
#len(atten_acc_nums)

# All PET-CTs include attenuation correction
imaging["attenuation_correction"] = np.where((imaging["ACCESSION_NUM"].isin(atten_acc_nums)) | (imaging["cardiac_test"]=="pet"), 1, 0)
#imaging["attenuation_correction"].value_counts()

# %%
# Extract date from string YYYY-MM-DD HH:MM
imaging["test_date"] = imaging["END_EXAM_DTTM"].str.extract(r"(\d{4}-\d{2}-\d{2})")
imaging["test_date"] = pd.to_datetime(imaging["test_date"])

# sort imaging by "PAT_MRN_ID", "test_date"
imaging = imaging.sort_values(by=['PAT_MRN_ID', 'test_date'])

# for each PAT_MRN_ID in imaging, keep the first row
imaging_first = imaging.drop_duplicates(subset=['PAT_MRN_ID'], keep='first')

# drop if cardiac_test is "angio"
imaging_first = imaging_first[imaging_first.cardiac_test != "angio"]

counts = imaging_first[["cardiac_test", "attenuation_correction"]].value_counts()
sorted_counts = counts.sort_index()
print(sorted_counts)

# %%
# Define pharmacological vs exercise
imaging_first["pharmacological"] = np.where(imaging_first["PROC_NAME"].isin(pharm), "pharm",
                                           np.where(imaging_first["PROC_NAME"].isin(exer), "exercise", "unknown"))     

# %%
# Confirm no exercise SPECT/echo same day
def check_tests(df):
    tests = df['cardiac_test'].values
    return ('ett' in tests) and ('spect' in tests or 'echo_stress' in tests)

result = imaging_first.groupby(['PAT_MRN_ID', 'test_date']).apply(check_tests)

len(result[result])

# %%
# Define counts of unique IDs to be included in the analysis
sel_ids = imaging_first["PAT_MRN_ID"].unique()
print("Number of unique patient IDs: ", len(sel_ids))

# %%
# create label/flag for abnormal imaging
imaging_first["abnormal_or_positive"] = np.where(imaging_first["ORDER_NARRATIVE"].str.contains("abnormal|positive", case=False, na=False), 1, 0)
imaging_first["abnormal_or_positive"].value_counts()

# %%
imaging_first["abnormal"] = np.where(imaging_first["ORDER_NARRATIVE"].str.contains("abnormal", case=False, na=False), 1, 0)
imaging_first["abnormal"].value_counts()

# %% [markdown]
# ## Pull encounters to link to inpatient vs outpatient

# %%
enc = pd.read_csv('/ehr-extract/2457789_CarDS_ASSIST_Hosp_Enc.txt', delimiter='\t', on_bad_lines='skip')

# %%
enc = enc[enc["PAT_MRN_ID"].isin(sel_ids)] # subsetting for selected patients
enc = enc[(enc["ED_YN"]==1)|(enc["INP_YN"]==1)] # subsetting for inpatient and ED patients

# %%
# Summarize reasons for inpatient/ED visits
enc["PRIMARY_RSN_FOR_VISIT_NAME"].value_counts(normalize=True)

# %%
# Merge the dataframes and get inpatient vs outpatient information
merged_df = pd.merge(imaging_first, enc, on='PAT_MRN_ID', how='left')

# Check if END_EXAM_DTTM is between HOSP_ADMSN_TIME and HOSP_DISCH_TIME
merged_df['inpatient'] = (
    (merged_df['END_EXAM_DTTM'] >= merged_df['HOSP_ADMSN_TIME']) & 
    (merged_df['END_EXAM_DTTM'] <= merged_df['HOSP_DISCH_TIME'])
).astype(int)

for col in ['END_EXAM_DTTM', 'HOSP_ADMSN_TIME', 'HOSP_DISCH_TIME']:
    merged_df.loc[merged_df[col].isna(), 'inpatient'] = float('nan')

# If you only want the original columns from imaging_first and the new inpatient column:
inpatient = merged_df[['PAT_MRN_ID', 'inpatient']]
inpatient = inpatient.groupby('PAT_MRN_ID')['inpatient'].max().reset_index()
inpatient["inpatient"].value_counts()

imaging_first = imaging_first.merge(inpatient, on="PAT_MRN_ID", how="left")

# %%
inpt = merged_df[merged_df["inpatient"]==1]
len(inpt)

# %%
inpt["PRIMARY_RSN_FOR_VISIT_NAME"].value_counts(normalize=True)

# %% [markdown]
# ## Plot trends

# %%
# Pool nuclear modalities together
imaging_first["cardiac_test_grp"] = np.where(imaging_first["cardiac_test"]=="spect", "nuc", 
                                             np.where(imaging_first["cardiac_test"]=="pet", "nuc", imaging_first["cardiac_test"]))

# %%
# Plot counts of cardiac_test by year (derived from "test_date")

import matplotlib.pyplot as plt

# Define years
imaging_first['year'] = imaging_first['test_date'].dt.year
grouped = imaging_first.groupby(['year', 'cardiac_test_grp']).size().unstack().fillna(0)
grouped.index = grouped.index.astype(int)
#grouped.index = pd.to_datetime(grouped.index, format='%Y')

# Rename the columns for the legend
rename_dict = {
    'ccta': 'CCTA',
    'cmr_stress': 'Stress CMR',
    'echo_stress': 'Stress Echo',
    'ett': 'ETT',
    'nuc': 'SPECT/PET MPI'
}
grouped.rename(columns=rename_dict, inplace=True)

grouped.plot(kind='bar', figsize=(10,6))
plt.title('Counts of cardiac_test by Year')
plt.ylabel('Count')
plt.xlabel('Year')
plt.xticks(rotation=0)
plt.gca().set_xticklabels(grouped.index)
plt.legend(title='Cardiac Test')
plt.tight_layout()
#plt.savefig('/path-to-figures/counts_cardiac_test_by_year_pet_spect.pdf')
plt.show()

# %%
# Show counts by years
grouped

# %%
# Plot percentage of cardiac_test by year

# Group by year and cardiac test to get counts
grouped = imaging_first.groupby(['year', 'cardiac_test_grp']).size().unstack().fillna(0)
grouped.index = grouped.index.astype(int)

rename_dict = {
    'ccta': 'CCTA',
    'cmr_stress': 'Stress CMR',
    'echo_stress': 'Stress Echo',
    'ett': 'ETT',
    'nuc': 'SPECT/PET MPI'
}
grouped.rename(columns=rename_dict, inplace=True)

# Convert the counts to percentages within each year
grouped_percentage = grouped.div(grouped.sum(axis=1), axis=0) * 100

# Plotting
grouped_percentage.plot(kind='bar', figsize=(10,6))
plt.title('Percentage of Each Cardiac Test by Year')
plt.ylabel('Percentage (%)')
plt.xlabel('Year')
plt.xticks(rotation=0)
plt.gca().set_xticklabels(grouped.index) 
plt.legend(title='Cardiac Test')
plt.tight_layout()
#plt.savefig('/path-to-figure/perc_cardiac_test_by_year_pet_spect.pdf')
plt.show()


# %%
# get tables of percentages
grouped_percentage

# %%
# Create cardiac_test_spec column
imaging_first["cardiac_test_spec"] = imaging_first["cardiac_test"]

# %%
# Write to a df
imaging_first.to_csv("/path-to-output/ASSIST_first_imaging_EHR_20240116.csv")

# %%
# Extract test date
test_date = imaging_first[["PAT_MRN_ID", "test_date"]]

# %% [markdown]
# ## Extract procedures

# %%
cpt = pd.read_csv('/ehr-extract/2457789_CarDS_ASSIST_Hosp_Enc_CPT.txt', delimiter='\t', on_bad_lines='skip')

# %%
# Subset selected ids to be included in this study
cpt = cpt[cpt["PAT_MRN_ID"].isin(sel_ids)]

# Identify cpt codes
cpt_cabg = ["33510", "33511", "33512", "33513", "33514", "33516", # using vein only
            "33517", "33518", "33519", "33521", "33522", "33523", # using venous and arterial grafts
            "33533", "33534", "33535", "33536"] # using arterial grafts

cpt_pci = ["92920", "92924", "92928", "92933",  "92937",  "92941", "92943",  "92975", "92977", # primary codes
           "92921", "92925", "92929", "92934", "92938", "92944", "92973", "92974", "92978", "92979"] # add on codes

# Identify cabg, pci
cpt["cabg"] = np.where(cpt["CPT_CODE"].isin(cpt_cabg), 1, 0)
cpt["pci"] = np.where(cpt["CPT_CODE"].isin(cpt_pci), 1, 0)
cpt = cpt[(cpt["pci"]==1)|(cpt["cabg"]==1)]

# %%
# Merge with dates and only keep procedures before
cpt_merged = cpt.merge(test_date, on="PAT_MRN_ID", how="left")
cpt_merged = cpt_merged[cpt_merged["CPT_DATE"]<cpt_merged["test_date"]]
len(cpt_merged)

# keep one row per patient
cpt_merged = cpt_merged.groupby("PAT_MRN_ID")[["cabg", "pci"]].max().reset_index()
cpt_merged.to_csv("/ehr-extract/ASSIST_cpt_pci_cabg_history.csv")

# %% [markdown]
# ## Load patient characteristics

# %% [markdown]
# #### Files are large so we will load them in chunks before editing them

# %%
def filter_large_file(filename, chunksize, ID_var, sel_ids):
    
    """
    Filters rows from a large file based on specific IDs.

    Args:
        filename (str): Path to the file.
        chunksize (int): Number of rows per chunk.
        ID_var (str): The column name that contains the IDs.
        sel_ids (list): List of IDs to filter by.

    Returns:
        DataFrame: Filtered data.
    """
    
    filtered_data = pd.DataFrame()

    # Iterate over the file in chunks
    for chunk in pd.read_csv(filename, delimiter='\t', chunksize=chunksize, on_bad_lines="skip"):
        filtered_chunk = chunk[chunk[ID_var].isin(sel_ids)]
        filtered_data = pd.concat([filtered_data, filtered_chunk])
    return filtered_data


# %%
# Dictionaries of ICD terms
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

t1dm_list = ['E10', 'E100', 'E101', 'E102','E103', 'E104', 'E105', 'E106', 'E107', 'E108',
             'E109', 'O240',
             
             '25001', '25011', '25021', '25091']

t2dm_list = ['E11', 'E110', 'E111', 'E112', 'E113', 'E114', 'E115', 'E116', 'E117', 'E118',
             'E119', 'O241',
             '25000', '25010', '25020', '25090']

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

pci_list = ["Z9861", "4582"]

cabg_list = ["Z951", "4581", "I2570", "I2571", "I2572", "I2573", "I2579", "I25810"]

pad_list = ['I702', 'I7020', 'I7021', 'I742', 'I743', 'I744'
            '4402', '4442']

stroke_list=['G45', 'G450', 'G451', 'G452', 'G453', 'G454', 'G458', 'G459', 'I63', 
             'I630', 'I631', 'I632', 'I633', 'I634', 'I635', 'I638', 'I639', 'I64',
             'I65', 'I650', 'I651', 'I652', 'I653', 'I658', 'I659', 'I66', 'I660',
             'I661', 'I662', 'I663', 'I664', 'I668', 'I669', 'I672', 'I693', 'I694',
             
             '433', '4330', '4331', '4332', '4333', '4338', '4339', '434', '4340',
             '4341', '4349', '435', '4359', '437', '4370', '4371']

active_list = ['Smoker', 'Smoking', 'Cigarette smoker', 'Smoking history', 'Current smoker'
               #... add extra terms here
               ]

former_list = ['Former smoker', 'Ex-smoker', 'Former cigarette smoker', 'Former smoker, stopped smoking in distant past'
               #... add extra terms here
               ]

pmh_dict = {'htn': htn_list, 'dm': dm_list, 't1dm': t1dm_list, 't2dm': t2dm_list,
            'ckd': ckd_list, 'heart_failure': heart_failure_list, 
            'ihd': ihd_list, 'pad': pad_list, 'stroke': stroke_list}

def add_dot(code):
    if len(code) > 3:
        return code[:3] + '.' + code[3:]
    else:
        return code

htn_list = [add_dot(code) for code in htn_list]
dm_list = [add_dot(code) for code in dm_list]
t1dm_list = [add_dot(code) for code in t1dm_list]
t2dm_list = [add_dot(code) for code in t2dm_list]
ckd_list = [add_dot(code) for code in ckd_list]
ami_list = [add_dot(code) for code in ami_list]
pci_list = [add_dot(code) for code in pci_list]
cabg_list = [add_dot(code) for code in cabg_list]
heart_failure_list = [add_dot(code) for code in heart_failure_list]
ihd_list = [add_dot(code) for code in ihd_list]
pad_list = [add_dot(code) for code in pad_list]
stroke_list = [add_dot(code) for code in stroke_list]

# %%
# get list of all files in a given directory
import os

directory_path = '/ehr-extract/2457789-CarDS-ASSIST'
file_list = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

#print(file_list)

# %% [markdown]
# ### General patient demographics

# %%
# Get patients demographics
pts = pd.read_csv('/ehr-extract/2457789_CarDS_ASSIST_Patients.txt', delimiter='\t', on_bad_lines='skip')
pts = pts[pts["PAT_MRN_ID"].isin(sel_ids)]
#len(pts)

# %%
# merge pts with test_date
pts_merged = pts.merge(test_date, on='PAT_MRN_ID', how='left')
pts_merged.head()

# %%
# Define CENSOR_DATE as DEATH_DATE, if nan, then 2023-08-03 
pts_merged["CENSOR_DATE"] = np.where(pts_merged["DEATH_DATE"].isnull(), "2023-08-03", pts_merged["DEATH_DATE"])

# Define TIME_TO_CENSOR as difference between CENSOR_DATE and test_date
pts_merged["TIME_TO_CENSOR"] = pd.to_datetime(pts_merged["CENSOR_DATE"]) - pd.to_datetime(pts_merged["test_date"])

# DEFINE DEATH
pts_merged["DEATH"] = np.where(pts_merged["DEATH_DATE"].isnull(), 0, 1)

# DEFINE AGE_TEST as age at test_date - BIRTH_DATE IN YEARS
pts_merged["AGE_TEST"] = (pd.to_datetime(pts_merged["test_date"]) - pd.to_datetime(pts_merged["BIRTH_DATE"])).dt.days/365

# Write to csv as elsewhere
pts_merged.to_csv("/ehr-extract/ASSIST_censoring.csv")


# %% [markdown]
# ### Problem list

# %%
# Load problem list
filtered_problem_list = filter_large_file('/ehr-extract/2457789_CarDS_ASSIST_Problem_List.txt', 10**6, 'PAT_MRN_ID', sel_ids)
filtered_problem_list = filtered_problem_list.merge(imaging_first, how='left', on='PAT_MRN_ID')
len(filtered_problem_list)

# %%
# create DATE_FIRST as the earliest date of DATE_OF_ENTRY and NOTED_DATE
filtered_problem_list['DX_DATE_FIRST'] = np.where(filtered_problem_list['NOTED_DATE']<filtered_problem_list['DATE_OF_ENTRY'], filtered_problem_list['NOTED_DATE'], filtered_problem_list['DATE_OF_ENTRY'])
filtered_problem_list['out_DATE_LAST'] = np.where(filtered_problem_list['RESOLVED_DATE'].isnull(), filtered_problem_list['DELETED_DATE'], filtered_problem_list['RESOLVED_DATE'])

# %% [markdown]
# #### a1) Comorbidities before

# %%
# select all diagnoses that are before the test date + 7 days
dx = filtered_problem_list.copy()
dx = dx[dx["DX_DATE_FIRST"] <= dx["test_date"] + pd.Timedelta(days=7)]
len(dx)

# %%
# Define diagnoses
# out
dx["HTN"] = np.where(dx["CURRENT_ICD10_LIST"].isin(htn_list) | dx["CURRENT_ICD9_LIST"].isin(htn_list), 1, 0)
htn_ids = dx[dx["HTN"] == 1]["PAT_MRN_ID"].unique()
print(dx["HTN"].value_counts())

# DM
dx["DM"] = np.where(dx["CURRENT_ICD10_LIST"].isin(dm_list) | dx["CURRENT_ICD9_LIST"].isin(dm_list), 1, 0)
dm_ids = dx[dx["DM"] == 1]["PAT_MRN_ID"].unique()
print(dx["DM"].value_counts())

# CKD
dx["CKD"] = np.where(dx["CURRENT_ICD10_LIST"].isin(ckd_list) | dx["CURRENT_ICD9_LIST"].isin(ckd_list), 1, 0)
ckd_ids = dx[dx["CKD"] == 1]["PAT_MRN_ID"].unique()
print(dx["CKD"].value_counts())

# HF
dx["HEART_FAILURE"] = np.where(dx["CURRENT_ICD10_LIST"].isin(heart_failure_list) | dx["CURRENT_ICD9_LIST"].isin(heart_failure_list), 1, 0)
hf_ids = dx[dx["HEART_FAILURE"] == 1]["PAT_MRN_ID"].unique()
print(dx["HEART_FAILURE"].value_counts())

# AMI
dx["AMI"] = np.where(dx["CURRENT_ICD10_LIST"].isin(ami_list) | dx["CURRENT_ICD9_LIST"].isin(ami_list), 1, 0)
ami_ids = dx[dx["AMI"] == 1]["PAT_MRN_ID"].unique()
print(dx["AMI"].value_counts())

# IHD
dx["IHD"] = np.where(dx["CURRENT_ICD10_LIST"].isin(ihd_list) | dx["CURRENT_ICD9_LIST"].isin(ihd_list), 1, 0)
ihd_ids = dx[dx["IHD"] == 1]["PAT_MRN_ID"].unique()
print(dx["IHD"].value_counts())

# PAD
dx["PAD"] = np.where(dx["CURRENT_ICD10_LIST"].isin(pad_list) | dx["CURRENT_ICD9_LIST"].isin(pad_list), 1, 0)
pad_ids = dx[dx["PAD"] == 1]["PAT_MRN_ID"].unique()
print(dx["PAD"].value_counts())

# STROKE
dx["STROKE"] = np.where(dx["CURRENT_ICD10_LIST"].isin(stroke_list) | dx["CURRENT_ICD9_LIST"].isin(stroke_list), 1, 0)
stroke_ids = dx[dx["STROKE"] == 1]["PAT_MRN_ID"].unique()
print(dx["STROKE"].value_counts())

# CABG
dx["CABG_ICD"] = np.where(dx["CURRENT_ICD10_LIST"].isin(cabg_list) | dx["CURRENT_ICD9_LIST"].isin(cabg_list), 1, 0)
stroke_ids = dx[dx["CABG_ICD"] == 1]["PAT_MRN_ID"].unique()
print(dx["CABG_ICD"].value_counts())

# PCI
dx["PCI_ICD"] = np.where(dx["CURRENT_ICD10_LIST"].isin(pci_list) | dx["CURRENT_ICD9_LIST"].isin(pci_list), 1, 0)
stroke_ids = dx[dx["PCI_ICD"] == 1]["PAT_MRN_ID"].unique()
print(dx["PCI_ICD"].value_counts())


# %%
dx_sorted = dx.sort_values(by=['PAT_MRN_ID', 'DX_DATE_FIRST'])

# Define an aggregation dictionary
agg_dict = {
    "HTN": 'max',
    "DM": 'max',
    "CKD": 'max',
    "HEART_FAILURE": 'max',
    "AMI": 'max',
    "IHD": 'max',
    "PAD": 'max',
    "CABG_ICD": 'max',
    "PCI_ICD": 'max',
    "STROKE": 'max',
    "DX_DATE_FIRST": 'first'  
}

# Group by 'PAT_MRN_ID' and aggregate
dx_aggregated = dx_sorted.groupby(['PAT_MRN_ID']).agg(agg_dict).reset_index()

# Get maximum and first DX date for each patient
for diag in ["HTN", "DM", "CKD", "HEART_FAILURE", "AMI", "IHD", "PAD", "STROKE"]:
    first_date_df = dx_sorted[dx_sorted[diag] == 1].groupby('PAT_MRN_ID')['DX_DATE_FIRST'].first().reset_index()
    first_date_df = first_date_df.rename(columns={'DX_DATE_FIRST': f'{diag}_DX_DATE_FIRST'})
    dx_aggregated = pd.merge(dx_aggregated, first_date_df, on='PAT_MRN_ID', how='left')

dx_aggregated = dx_aggregated[dx_aggregated["PAT_MRN_ID"].isin(sel_ids)]
dx_aggregated.to_csv("/ehr-extract/ASSIST_comorbidities_dates_20231102.csv", index=False)


# %%
len(dx_aggregated)

# %% [markdown]
# ### a2) f/u for stroke or ami

# %%
# select all diagnoses that occurred >7 days after the test date
out = filtered_problem_list.copy()
out = out[out["DX_DATE_FIRST"] > out["test_date"] + pd.Timedelta(days=7)]
len(out)

# %%
# Define diagnoses

# HF
out["FU_HEART_FAILURE"] = np.where(out["CURRENT_ICD10_LIST"].isin(heart_failure_list) | out["CURRENT_ICD9_LIST"].isin(heart_failure_list), 1, 0)
hf_ids = out[out["FU_HEART_FAILURE"] == 1]["PAT_MRN_ID"].unique()
print(out["FU_HEART_FAILURE"].value_counts())

# AMI
out["FU_AMI"] = np.where(out["CURRENT_ICD10_LIST"].isin(ami_list) | out["CURRENT_ICD9_LIST"].isin(ami_list), 1, 0)
ami_ids = out[out["FU_AMI"] == 1]["PAT_MRN_ID"].unique()
print(out["FU_AMI"].value_counts())

# STROKE
out["FU_STROKE"] = np.where(out["CURRENT_ICD10_LIST"].isin(stroke_list) | out["CURRENT_ICD9_LIST"].isin(stroke_list), 1, 0)
stroke_ids = out[out["FU_STROKE"] == 1]["PAT_MRN_ID"].unique()
print(out["FU_STROKE"].value_counts())


# %%
out_sorted = out.sort_values(by=['PAT_MRN_ID', 'DX_DATE_FIRST'])

# Define an aggregation dictionary
agg_dict = {
    "FU_HEART_FAILURE": 'max',
    "FU_AMI": 'max',
    "FU_STROKE": 'max',
    "DX_DATE_FIRST": 'first'
}

# Group by 'PAT_MRN_ID' and aggregate
out_aggregated = out_sorted.groupby(['PAT_MRN_ID']).agg(agg_dict).reset_index()

# Get maximum and first DX date for each patient
for diag in ["FU_HEART_FAILURE", "FU_AMI", "FU_STROKE"]:
    first_date_df = out_sorted[out_sorted[diag] == 1].groupby('PAT_MRN_ID')['DX_DATE_FIRST'].first().reset_index()
    first_date_df = first_date_df.rename(columns={'DX_DATE_FIRST': f'{diag}_DX_DATE_FIRST'})
    out_aggregated = pd.merge(out_aggregated, first_date_df, on='PAT_MRN_ID', how='left')

out_aggregated = out_aggregated[out_aggregated["PAT_MRN_ID"].isin(sel_ids)]
out_aggregated.to_csv("/ehr-extract/ASSIST_outcome_dates.csv", index=False)
out_aggregated.head()


# %% [markdown]
# #### b) Smoking

# %%
smoking = filtered_problem_list.copy()

# %%
all_smoking = smoking[smoking["DX_NAME"].str.contains("tobacco|smok|cigar|nicotine|nicotin")]["DX_NAME"].unique()
#print(all_smoking)

active_list = [
       'Personal history of smoking',
       'Nicotine dependence, cigarettes, with unspecified nicotine-induced disorders',
       'History of smoking', 
       'Current tobacco use',
       'Continuous dependence on cigarette smoking'
       #... add more
       ]

former_list = [
       'Former smoker',
       'Ex-smoker',
       'Former cigarette smoker', 'Ex-smoker for more than 1 year'
       #... add more
       ]


no_smoking_list = [
       'Nonsmoker'
       #... add more
       ]


# %%
# SMOKERS
smoking["SMOKERS"] = np.where(smoking["DX_NAME"].isin(active_list)| smoking["DX_NAME"].isin(former_list), 1, 0)
smoking = smoking[smoking["SMOKERS"]==1]
smoker_ids = smoking[smoking["SMOKERS"] == 1]["PAT_MRN_ID"].unique()
#print(smoking["SMOKERS"].value_counts())

# %%
# Define date of quitting smoking
smoking["DATE_SMOKE_QUIT"] = np.where(smoking["DX_NAME"].isin(active_list), smoking["DX_DATE_LAST"],
                                                        np.where(smoking["DX_NAME"].isin(former_list), smoking["DX_DATE_FIRST"], '2099-12-31'))

# %%
# Define active and past smoker ids based on quit date relative to test_date
active_smoker_ids = smoking[(smoking["DX_NAME"].isin(active_list)) & (smoking["DATE_SMOKE_QUIT"]>smoking["test_date"])]["PAT_MRN_ID"].unique()
len(active_smoker_ids)

# Define active and past smoker ids based on quit date relative to test_date
former_smoker_ids = smoking[~smoking["PAT_MRN_ID"].isin(active_smoker_ids)]["PAT_MRN_ID"].unique()
len(former_smoker_ids)

#print("In summary, there are {} active smokers and {} former smokers in the dataset.".format(len(active_smoker_ids), len(former_smoker_ids)))

# %%
smoking["SMOKER_ACTIVE"] = np.where(smoking["PAT_MRN_ID"].isin(active_smoker_ids), 1, 0)
smoking["SMOKER_FORMER"] = np.where(smoking["PAT_MRN_ID"].isin(former_smoker_ids), 1, 0)
smoking = smoking[["PAT_MRN_ID", "DATE_SMOKE_QUIT", "SMOKER_ACTIVE", "SMOKER_FORMER"]]
smoking.to_csv("/ehr-extract/ASSIST_smoking.csv", index=False)

# %% [markdown]
# ## Labs

# %%
ch_vals = ['BKR CHOLESTEROL', 'CHOLESTEROL, TOTAL', 'CHOLESTEROL', 'POC CHOLESTEROL', 'CHOLESTEROL (ABSTRACTED)']
hdl_vals = ['BKR HDL CHOLESTEROL', 'HDL CHOLESTEROL', 'HIGH DENSITY CHOLESTEROL', 'POC HIGH DENSITY CHOLESTEROL', 'HDL (ABSTRACTED)']

# %%
import re
import os

def extract_float(s):
    result = re.findall(r"(\d+\.\d+)", s)
    if result:
        return float(result[0])
    return None

directory = "/ehr-extract/Labs"
lab_files = [
    "2457789_CarDS_ASSIST_Outpatient_Enc_Labs_1.txt", 
    "2457789_CarDS_ASSIST_Outpatient_Enc_Labs_2.txt",
    "2457789_CarDS_ASSIST_Hosp_Enc_Labs_5.txt", # all lipid parameters are in the _5.txt file
    "2457789_CarDS_ASSIST_Hosp_Enc_Labs_1.txt", "2457789_CarDS_ASSIST_Hosp_Enc_Labs_2.txt", 
    "2457789_CarDS_ASSIST_Hosp_Enc_Labs_3.txt", "2457789_CarDS_ASSIST_Hosp_Enc_Labs_4.txt"
]

# Placeholder for the final result
final_df = None

for file in lab_files:
    path = os.path.join(directory, file)
    chunked_df = pd.read_csv(path, chunksize=10**6, delimiter='\t', on_bad_lines='skip')
    
    for chunk in chunked_df:
        chunk = chunk[ ((chunk['COMPONENT_NAME'].isin(ch_vals))|chunk['COMPONENT_NAME'].isin(hdl_vals)) & chunk["PAT_MRN_ID"].isin(sel_ids)]
        #chunk = chunk[chunk['COMPONENT_NAME'].str.contains("chol", case=False, na=False)]

        if final_df is None:
            final_df = chunk
        else:
            final_df = final_df.append(chunk)


# %%
# merge with test_date
labs = final_df.merge(imaging_first, how='left', on='PAT_MRN_ID')

# select only labs done before or on the test date
labs = labs[labs["SPECIMN_TAKEN_DATE"] <= (labs["test_date"]+pd.Timedelta(days=0))]

# extract float from ORD_NUM_VALUE
labs['cholesterol'] = labs[labs['COMPONENT_NAME'].isin(['BKR CHOLESTEROL', 'CHOLESTEROL, TOTAL', 'CHOLESTEROL',
                                                        'POC CHOLESTEROL', 'CHOLESTEROL (ABSTRACTED)'])]['ORD_NUM_VALUE'].replace({'TNP': np.nan, '<100': 100, '<50': 50}).astype(float)
labs['hdl'] = labs[labs['COMPONENT_NAME'].isin(['BKR HDL CHOLESTEROL', 'HDL CHOLESTEROL', 'HIGH DENSITY CHOLESTEROL',
                                                'POC HIGH DENSITY CHOLESTEROL', 'HDL (ABSTRACTED)'])]['ORD_NUM_VALUE'].replace({'TNP': np.nan}).astype(float)

# only keep the columns where cholesterol OR hdl are >0
labs = labs[(labs['cholesterol']>0) | (labs['hdl']>0)]

# %%
# Sort the dataframe by 'PAT_MRN_ID' and 'SPECIMN_TAKEN_DATE' so that the last record for each patient is the latest
labs = labs.sort_values(['PAT_MRN_ID', 'SPECIMN_TAKEN_DATE'])

# Group by 'PAT_MRN_ID' and get the last 'cholesterol' and 'hdl' and their respective dates
labs_final = labs.groupby('PAT_MRN_ID').agg(
    cholesterol=pd.NamedAgg(column='cholesterol', aggfunc='last'),
    cholesterol_date=pd.NamedAgg(column='SPECIMN_TAKEN_DATE', aggfunc='last'),
    hdl=pd.NamedAgg(column='hdl', aggfunc='last'),
    hdl_date=pd.NamedAgg(column='SPECIMN_TAKEN_DATE', aggfunc='last')
).reset_index()

# Remove rows where both cholesterol and hdl are NaN
labs_final = labs_final.dropna(subset=['cholesterol', 'hdl'], how='all')

len(labs_final) 


# %%
# Write labs to csv file as above
labs_final.to_csv("/ehr-extract/ASSIST_labs.csv", index=False)

# %% [markdown]
# ### BMI/vitals

# %%
vitals_inpt = pd.read_csv('/ehr-extract/2457789_CarDS_ASSIST_Outpatient_Enc_Flo_Vitals.txt',  delimiter='\t', on_bad_lines='skip')

# %%
vitals_outpt = pd.read_csv('/ehr-extract/2457789_CarDS_ASSIST_Hosp_Enc_Flo_Vitals.txt',  delimiter='\t', on_bad_lines='skip')

# %%
vitals = pd.concat([vitals_inpt, vitals_outpt], axis=0, ignore_index=True)

# %%
vitals = vitals[vitals["PAT_MRN_ID"].isin(sel_ids)]

# %%
vitals["DISP_NAME"].value_counts()

# %%
bmi = vitals[vitals["DISP_NAME"].isin(["Height", "Weight"])]
len(bmi)

# %%
bp = vitals[vitals["DISP_NAME"].isin(["BP"])]
len(bp)

# %%
bmi = bmi.merge(test_date, on="PAT_MRN_ID", how="left")
bmi = bmi[bmi["RECORDED_TIME"]<=bmi["test_date"]]
len(bmi)

# %%
bp = bp.merge(test_date, on="PAT_MRN_ID", how="left")
bp = bp[bp["RECORDED_TIME"]<=bp["test_date"]]
len(bp)

# %%
wt = bmi[bmi["DISP_NAME"]=="Weight"]
ht = bmi[bmi["DISP_NAME"]=="Height"] 

# %%
# Split the MEAS_VALUE column
bp[['bp1', 'bp2']] = bp['MEAS_VALUE'].str.split('/', expand=True)

# Convert to numeric
bp['bp1'] = pd.to_numeric(bp['bp1'], errors='coerce')
bp['bp2'] = pd.to_numeric(bp['bp2'], errors='coerce')

# Create sbp and dbp columns
bp['sbp'] = np.where(bp['bp1'] > bp['bp2'], bp['bp1'], bp['bp2'])
bp['dbp'] = np.where(bp['bp1'] < bp['bp2'], bp['bp1'], bp['bp2'])

# %%
# sort wt, then ht by "PAT_MRN_ID", then "RECORDED_TIME" and keep the last record for each PAT_MRN_ID
wt = wt.sort_values(by=['PAT_MRN_ID', 'RECORDED_TIME'], ascending=[True, False])
wt = wt.drop_duplicates(subset=['PAT_MRN_ID'], keep='first')

ht = ht.sort_values(by=['PAT_MRN_ID', 'RECORDED_TIME'], ascending=[True, False])
ht = ht.drop_duplicates(subset=['PAT_MRN_ID'], keep='first')

# %%
bp = bp.sort_values(by=['PAT_MRN_ID', 'RECORDED_TIME'], ascending=[True, False])
bp = bp.drop_duplicates(subset=['PAT_MRN_ID'], keep='first')

# %%
# merge ht, wt on "PAT_MRN_ID" but keep all rows from both df, for vars add _ht or _wt as suffix
bmi_joined = pd.merge(ht, wt, on="PAT_MRN_ID", how="outer", suffixes=("_ht", "_wt"))
bmi_joined["bmi"] = ((float(bmi_joined["MEAS_VALUE_wt"])*0.0625) / (float(bmi_joined["MEAS_VALUE_ht"])**2)) * 703
bmi_joined["bmi"].describe()

# %%
# merge ht, wt on "PAT_MRN_ID" but keep all rows from both df, for vars add _ht or _wt as suffix
bmi_joined = pd.merge(ht, wt, on="PAT_MRN_ID", how="outer", suffixes=("_ht", "_wt"))

# Compute BMI element-wise
bmi_joined["bmi"] = ((bmi_joined["MEAS_VALUE_wt"].astype(float)*0.0625) / (bmi_joined["MEAS_VALUE_ht"].astype(float) ** 2)) * 703

# Get the description of the BMI column
desc = bmi_joined["bmi"].describe()
print(desc)

# write bmi_joined to csv
#bmi_joined.to_csv("/ehr-extract/bmi_joined.csv", index=False)


# %%
bp_sel = bp[["PAT_MRN_ID", "sbp", "dbp"]]

# if sbp < 50 or sbp > 250 or dbp < 30 or dbp > 150, then set to NaN
bp_sel["sbp"] = np.where((bp_sel["sbp"] < 50) | (bp_sel["sbp"] > 250), np.nan, bp_sel["sbp"])
bp_sel["dbp"] = np.where((bp_sel["dbp"] < 30) | (bp_sel["dbp"] > 150), np.nan, bp_sel["dbp"])

bp_sel.to_csv("/ehr-extract/bp_sel.csv", index=False)

# %% [markdown]
# ### Medications

# %%
meds = pd.read_csv('/ehr-extract/2457789_CarDS_ASSIST_Meds.txt',  delimiter='\t', on_bad_lines='skip')
meds_merged = meds.merge(test_date, on='PAT_MRN_ID', how='left')
#print(meds_merged["PAT_MRN_ID"].nunique())
#print(meds_merged["PHARM_CLASS"].value_counts())

# %%
meds_merged = meds_merged[meds_merged["PAT_MRN_ID"].isin(sel_ids)]
len(meds_merged)

# %%
# keep only meds ordered within 2 years of the study
meds_merged_sel = meds_merged[(meds_merged['ORDER_INST']<meds_merged["test_date"]) & (meds_merged['ORDER_INST']>=meds_merged["test_date"]-pd.Timedelta(days=365*2))]

# separate subset for medications within 1 year of the study
meds_merged_after = meds_merged[(meds_merged['ORDER_INST']>meds_merged["test_date"]) & (meds_merged['ORDER_INST']<=meds_merged["test_date"]+pd.Timedelta(days=365))]

# %%
antihypertensives = ["BETA-ADRENERGIC BLOCKING AGENTS",
    "ALPHA/BETA-ADRENERGIC BLOCKING AGENTS",
    "ALPHA-ADRENERGIC BLOCKING AGENTS",
    "ALPHA-ADRENERGIC BLOCKING AGENT/THIAZIDE COMB",
    "BETA-BLOCKERS AND THIAZIDE,THIAZIDE-LIKE DIURETICS",
    "CALCIUM CHANNEL BLOCKING AGENTS",
    "ANTIHYPERLIPID- HMG-COA RI-CALCIUM CHANNEL BLOCKER",
    "RENIN INHIB,DIRECT-CALCIUM CHANNEL BLOCKR-THIAZIDE",
    "RENIN INHIBITOR,DIRECT AND CALCIUM CHANNEL BLOCKER",
    "ANTIHYPERTENSIVES, ACE INHIBITORS",
    "ACE INHIBITOR-THIAZIDE OR THIAZIDE-LIKE DIURETIC",
    "ACE INHIBITOR-CALCIUM CHANNEL BLOCKER COMBINATION",
    "ANTIHYPERTENSIVES, ANGIOTENSIN RECEPTOR ANTAGONIST",
    "ANGIOTENSIN RECEPTOR ANTAG.-THIAZIDE DIURETIC COMB",
    "ANGIOTENSIN RECEPT-NEPRILYSIN INHIBITOR COMB(ARNI)",
    "ANGIOTENSIN RECEPTOR BLOCKR-CALCIUM CHANNEL BLOCKR",
    "RENIN INHIBITOR,DIRECT-ANGIOTENSIN RECEPTR ANTAGON",
    "ANGIOTENSIN II RECEPTOR BLOCKER-BETA BLOCKER COMB.",
    "LOOP DIURETICS",
    "THIAZIDE AND RELATED DIURETICS",
    "RENIN INHIBITOR,DIRECT AND THIAZIDE DIURETIC COMB",
    "POTASSIUM SPARING DIURETICS",
    "POTASSIUM SPARING DIURETICS IN COMBINATION",
    "ANTIHYPERTENSIVES, VASODILATORS",
    "ANTIHYPERTENSIVES, SYMPATHOLYTIC",
    "ANTIHYPERTENSIVES, MISCELLANEOUS",
    "ANTIHYPERTENSIVES, GANGLIONIC BLOCKERS",
    "ANTIHYPERTENSIVES,ACE INHIBITOR/DIETARY SUPP.COMB."
]

# %%
meds_merged_sel["antihypertensive"] = np.where(meds_merged_sel["PHARM_CLASS"].isin(antihypertensives), 1, 0)
meds_merged_sel["antihypertensive"].value_counts()

# %%
meds_merged_after["antihypertensive"] = np.where(meds_merged_after["PHARM_CLASS"].isin(antihypertensives), 1, 0)
meds_merged_after["antihypertensive"].value_counts()

# %%
meds_merged_sel['statin'] = np.where(meds_merged_sel['PHARM_CLASS']=='ANTIHYPERLIPIDEMIC-HMGCOA REDUCTASE INHIB(STATINS)', 1, 0)
meds_merged_sel['antiplatelet'] = np.where(meds_merged_sel['PHARM_CLASS']=='PLATELET AGGREGATION INHIBITORS', 1, 0)
meds_merged_sel['beta_blocker'] = np.where(meds_merged_sel['PHARM_CLASS'].isin(['BETA-ADRENERGIC BLOCKING AGENTS', 'ALPHA/BETA-ADRENERGIC BLOCKING AGENTS','BETA-BLOCKERS AND THIAZIDE,THIAZIDE-LIKE DIURETICS']), 1, 0)
meds_merged_sel['statin'] = meds_merged_sel.groupby(by='PAT_MRN_ID')['statin'].transform(max)
meds_merged_sel['antiplatelet'] = meds_merged_sel.groupby(by='PAT_MRN_ID')['antiplatelet'].transform(max)
meds_merged_sel['beta_blocker'] = meds_merged_sel.groupby(by='PAT_MRN_ID')['beta_blocker'].transform(max)
len(meds_merged_sel)


# %%
meds_merged_after['statin_'] = np.where(meds_merged_after['PHARM_CLASS']=='ANTIHYPERLIPIDEMIC-HMGCOA REDUCTASE INHIB(STATINS)', 1, 0)
meds_merged_after['antiplatelet'] = np.where(meds_merged_after['PHARM_CLASS']=='PLATELET AGGREGATION INHIBITORS', 1, 0)
meds_merged_after['beta_blocker'] = np.where(meds_merged_after['PHARM_CLASS'].isin(['BETA-ADRENERGIC BLOCKING AGENTS', 'ALPHA/BETA-ADRENERGIC BLOCKING AGENTS','BETA-BLOCKERS AND THIAZIDE,THIAZIDE-LIKE DIURETICS']), 1, 0)
meds_merged_after['statin'] = meds_merged_after.groupby(by='PAT_MRN_ID')['statin'].transform(max)
meds_merged_after['antiplatelet'] = meds_merged_after.groupby(by='PAT_MRN_ID')['antiplatelet'].transform(max)
meds_merged_after['beta_blocker'] = meds_merged_after.groupby(by='PAT_MRN_ID')['beta_blocker'].transform(max)
len(meds_merged_after)

# %%
meds_merged_sel = meds_merged_sel[["PAT_MRN_ID", "statin", "antiplatelet", "beta_blocker", "antihypertensive"]]
meds_merged_sel = meds_merged_sel.groupby(by='PAT_MRN_ID').max().reset_index()

print("In total, there are", len(meds_merged_sel), "patients with meds data. Of these, ", meds_merged_sel["statin"].sum(), 
      "are on statins, ", meds_merged_sel["antiplatelet"].sum(), "are on antiplatelets,", meds_merged_sel["beta_blocker"].sum(), 
      "are on beta blockers, and ", meds_merged_sel["antihypertensive"].sum(), "are on antihypertensives.")

# %%
meds_merged_after = meds_merged_after[["PAT_MRN_ID", "statin", "antiplatelet", "beta_blocker", "antihypertensive"]]
meds_merged_after = meds_merged_after.groupby(by='PAT_MRN_ID').max().reset_index()

print("In total, there are", len(meds_merged_after), "patients with meds data. Of these, ", meds_merged_after["statin"].sum(),
      "are on statins, ", meds_merged_after["antiplatelet"].sum(), "are on antiplatelets,", meds_merged_after["beta_blocker"].sum(),
      "are on beta blockers, and ", meds_merged_after["antihypertensive"].sum(), "have antihypertesive prescriptions within 12 months AFTER the study.")

# %%
# Add suffix to column names
meds_merged_after["antiplatelet_within1y"] = meds_merged_after["antiplatelet"]
meds_merged_after["statin_within1y"] = meds_merged_after["statin"]
meds_merged_after["beta_blocker_within1y"] = meds_merged_after["beta_blocker"]
meds_merged_after["antihypertensive_within1y"] = meds_merged_after["antihypertensive"]
meds_merged_after = meds_merged_after[["PAT_MRN_ID", "antiplatelet_within1y", "statin_within1y", "beta_blocker_within1y", "antihypertensive_within1y"]]

# %%
# Merge original data with all meds
meds_final = test_date.merge(meds_merged_sel, on="PAT_MRN_ID", how="left")
meds_final = meds_final[["PAT_MRN_ID", "statin", "antiplatelet", "beta_blocker", "antihypertensive"]]

meds_final = meds_final.merge(meds_merged_after, on="PAT_MRN_ID", how="left")

# %%
meds_final = meds_final.fillna(0)

# %%
# New med variables
meds_final["antiplatelet_new"] = np.where((meds_final["antiplatelet_within1y"]==1) & (meds_final["antiplatelet"]==0), 1, 0)
meds_final["statin_new"] = np.where((meds_final["statin_within1y"]==1) & (meds_final["statin"]==0), 1, 0)
meds_final["beta_blocker_new"] = np.where((meds_final["beta_blocker_within1y"]==1) & (meds_final["beta_blocker"]==0), 1, 0)
meds_final["antihypertensive_new"] = np.where((meds_final["antihypertensive_within1y"]==1) & (meds_final["antihypertensive"]==0), 1, 0)

print("In total patients on new meds include", meds_final["antiplatelet_new"].sum(), "on antiplatelets,", meds_final["statin_new"].sum(), "on statins,", meds_final["beta_blocker_new"].sum(), "on beta blockers, and", meds_final["antihypertensive_new"].sum(), "on antihypertensives.")

# %%
# write csv as above
meds_final.to_csv("/ehr-extract/ASSIST_meds_with_antihypertensives_and_new_meds.csv", index=False)

# %% [markdown]
# ## Defining primary outcomes
# #### We will define AMI based on encounters (ED/inpatient) + primary ICD codes for AMI

# %%
# Load previous file, identify study participants
assist_ehr = pd.read_csv("/ehr-extract/ASSIST_Yale_final_20240116.csv")
sel_ids = assist_ehr["pat_mrn_id"].unique()

# %%
# Extract test date from above
imaging_first = imaging_first[["PAT_MRN_ID", "test_date"]]
imaging_first.columns = [col.lower() for col in imaging_first.columns]

# %%
# Use filter function from above - load the encounter file
filtered_inpt_enc = filter_large_file('/ehr-extract/2457789_CarDS_ASSIST_Hosp_Enc.txt', 10**6, 'PAT_MRN_ID', sel_ids)
filtered_inpt_enc_inpted = filtered_inpt_enc[filtered_inpt_enc["ACCT_BASECLS_HA"].isin(["Emergency", "Inpatient"])] # subset for inpt and ED visits
filtered_inpt_enc_inpted_csn = filtered_inpt_enc_inpted["PAT_ENC_CSN_ID"].unique() # get the corresponding CSNs

# %%
# Now move the encounter diagnoses
filtered_inpt_icd = filter_large_file('/ehr-extract/2457789_CarDS_ASSIST_Hosp_Enc_DX.txt', 10**6, 'PAT_MRN_ID', sel_ids) 
filtered_inpt_icd["AMI"] = np.where(filtered_inpt_icd["CURRENT_ICD10_LIST"].isin(ami_list) | filtered_inpt_icd["CURRENT_ICD9_LIST"].isin(ami_list), 1, 0) # define AMI
filtered_inpt_icd_ami = filtered_inpt_icd[filtered_inpt_icd["AMI"]==1] # subset for AMI
filtered_inpt_icd_ami.columns = [col.lower() for col in filtered_inpt_icd_ami.columns] # lowercase to make compatible with original df
filtered_inpt_icd_ami = filtered_inpt_icd_ami[filtered_inpt_icd_ami["pat_enc_csn_id"].isin(filtered_inpt_enc_inpted_csn)] # keep only events corresponding to the above encounters


# %%
# Merge with the imaging dates
combined = imaging_first.merge(filtered_inpt_icd_ami, on="pat_mrn_id", how="left") # merge with the test dates
combined_sel = combined[combined["ami"]==1] # only keep ami (maybe duplicated from above, but making sure)
combined_sel["test_date"] = pd.to_datetime(combined_sel["test_date"]) # get dates
combined_sel["calc_dx_date"] = pd.to_datetime(combined_sel["calc_dx_date"])
combined_sel["delta"] = combined_sel["calc_dx_date"] - combined_sel["test_date"] # get time difference
filtered_df = combined_sel[combined_sel["delta"] > pd.Timedelta(days=7)] # exclude any events within 7 days
first_occurrence_indices = filtered_df.groupby("pat_mrn_id")["calc_dx_date"].idxmin() # filter by first occurrence
filtered_df = filtered_df.loc[first_occurrence_indices] 
filtered_df["pat_mrn_id"].nunique()

# %%
# if you want you can subset for primary codes - not doing here since this is AMI
filtered_df_prim = filtered_df[filtered_df["primary_yn"]=="Y"]
filtered_df_prim["pat_mrn_id"].nunique()

# %%
# Define updated endpoints (AMI and 2-COMPONENT MACE OF DEATH/AMI)
filtered_df["ami2"] = filtered_df["ami"]
filtered_df["time_to_ami2"] = filtered_df["delta"].dt.days
filtered_df = filtered_df[["pat_mrn_id", "test_date", "ami2", "time_to_ami2"]]

# %%
# Merge with the original df
assist_ehr_upd = assist_ehr.merge(filtered_df, on="pat_mrn_id", how="left")

# %%
# fill data for those with no ami2
assist_ehr_upd["ami2"].fillna(0, inplace=True)
assist_ehr_upd["time_to_ami2"].fillna(assist_ehr_upd["time_to_censor"], inplace=True)

# %%
# define new MACE2_2
assist_ehr_upd["MACE2_2"] = np.where((assist_ehr_upd["ami2"]==1) | (assist_ehr_upd["death"]==1), 1, 0)
assist_ehr_upd["time_to_MACE2_2"] = np.where(assist_ehr_upd["ami2"]==1, assist_ehr_upd["time_to_ami2"], assist_ehr_upd["time_to_censor"])

# %%
# Write to new file
assist_ehr_upd.to_csv('/ehr-extract/ASSIST_Yale_final_20240116b.csv')