# the imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, PolynomialFeatures

df = pd.read_csv('../data/cleaned_data.csv')

# to avoid excess column dummification in the next step it is necessary to numerically categorize the quality columns as per example in overall quality column (but keeping it from 0 (not available) to 5 (excelent))

numer_cols = ['exter_qual', 'exter_cond', 'bsmt_qual', 'bsmt_cond', 'bsmt_exposure', 
              'heating_qc', 'kitchen_qual', 'fireplace_qu', 'garage_qual','garage_cond',
              'pool_qc']

def numerizer(val):
    if val == 'Ex':
        return 5
    elif val == 'Gd':
        return 4
    elif val == 'TA' or val == 'Av':
        return 3
    elif val == 'Fa' or val == 'Mn':
        return 2
    elif val == 'Po' or val == 'No':
        return 1
    else:
        return 0

for col in numer_cols:
    df[col] = df[col].map(numerizer)

df.to_csv('../data/cleaned_numerized_data.csv', index=False)

# ### Adding interaction columns

df['overall_qual_cond'] = df['overall_qual'] * df['overall_cond']
df['exter_qual_cond'] = df['exter_qual'] * df['exter_cond']
df['bsmt_qual_cond'] = df['bsmt_qual'] * df['bsmt_cond']
df['bsmt_qual_cond_exposure'] = df['bsmt_qual'] * df['bsmt_cond'] * df['bsmt_exposure']
df['garage_qual_cond'] = df['garage_qual'] * df['garage_cond']

df.to_csv('../data/numer_interact_data.csv', index=False)

# ### Dummyfiyng all the categorical columns
# This step is praparatory for further interaction analysis, and seeing how columns interact with each other.

dummy_cols = ['ms_subclass', 'ms_zoning', 'street', 'alley', 'lot_shape',
              'land_contour', 'utilities', 'lot_config', 'land_slope', 'neighborhood',
              'condition_1', 'condition_2', 'bldg_type', 'house_style', 'roof_style',
              'roof_matl', 'exterior_1st', 'exterior_2nd', 'mas_vnr_type',
              'foundation', 'bsmtfin_type_1', 'bsmtfin_type_2', 'heating', 'central_air', 
              'electrical', 'functional', 'garage_type', 'garage_finish', 'paved_drive',
              'fence', 'misc_feature', 'sale_type']
df = pd.get_dummies(df, columns = dummy_cols)

df.to_csv('../data/dummy_numerize_interact_data.csv', index=False)




