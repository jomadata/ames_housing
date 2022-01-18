
# The imports
import numpy as np
import pandas as pd

df = pd.read_csv('../data/train.csv')

#camel_casing the column names
col_dict = {}
for col in df.columns:
    col_dict[col] = col.lower().replace(' ', '_')
df.rename(columns = col_dict, inplace=True)

# Will try to fill with average of plots with same area (only ~15% of values are np.nan)
# Second loop will fill the lot frontage NA's with generalized average as not always there is a local average
lot_area = sorted(list(df['lot_area'].value_counts().index))
missing_lots = []
for lot in lot_area:
    if len(df[(df['lot_area'] == lot) & (df['lot_frontage'].isna())]) != 0:
        try:
            local_av = round(df[(df['lot_area'] == lot) & (df['lot_frontage'].notna())]['lot_frontage'].mean())
            df[df['lot_area'] == lot]['lot_frontage'].fillna(local_av, inplace=True)
        except:
            missing_lots.append(lot)
for lt in missing_lots:
        gen_av = round(df[df['lot_frontage'].notna()]['lot_frontage'].mean())
        df['lot_frontage'].fillna(gen_av, inplace=True)

# according to dictionary it should be NA and not null values
df['alley'].fillna('no_alley', inplace=True)

# under the assumption that null values should be None
df['mas_vnr_type'].fillna('no_masonry', inplace=True)
def mas(val):
    return 'no_masonry' if val=='None' else val
df['mas_vnr_type'] = df['mas_vnr_type'].map(mas) 

# under the assumption that Null values should equal to 0 will fill na
df['mas_vnr_area'].fillna(0, inplace=True)

# under the assumption that Null values should equal to NA will fill na
df['bsmt_qual'].fillna('no_basement', inplace=True)
df['bsmt_cond'].fillna('no_basement', inplace=True)
df['bsmt_exposure'].fillna('no_basement', inplace=True)
df['bsmtfin_type_1'].fillna('no_basement', inplace=True)
df['bsmtfin_type_2'].fillna('no_basement', inplace=True)

# under the assumption that null values should equal to 0 will fill na
df['bsmtfin_sf_1'].fillna(0, inplace=True)
df['bsmtfin_sf_2'].fillna(0, inplace=True)

#under the assumption that null values should equal to 0 will fill na
df['bsmt_unf_sf'].fillna(0, inplace=True)
df['total_bsmt_sf'].fillna(0, inplace=True)

#under the assumption that null values should equal to 0 will fill na
df['bsmt_full_bath'].fillna(0, inplace=True)
df['bsmt_half_bath'].fillna(0, inplace=True)
df['bsmt_full_bath'].isna().sum(), df['bsmt_half_bath'].isna().sum()

#under the assumption that null values should equal to 0 will fill na, according to dictionary
df['fireplace_qu'].fillna(0, inplace=True)

#under the assumption that null values should equal to NA will fill na, according to dictionary
df['garage_type'].fillna('no_garage', inplace=True)


# can not dummify the garage_yr_blt column because of too many dummy columns, can not fill with
#'NA' as it is vital numeric data, deleting 5% of data is not an option, last resort
# is to fill NaN values with zeros
df['garage_yr_blt'].fillna(0, inplace=True)
df['garage_yr_blt'].isna().sum()

# under assumption that null values are NA values from the dictionary
df['garage_finish'].fillna('no_garage', inplace=True)
df['garage_qual'].fillna('no_garage', inplace=True)
df['garage_finish'].fillna('no_garage', inplace=True)
df['garage_cond'].fillna('no_garage', inplace=True)
df['garage_finish'].isna().sum(), df['garage_qual'].isna().sum(), df['garage_cond'].isna().sum()

# under the assumption that non existent garage will have 0 sq_ft of area
df['garage_cars'].fillna(0, inplace=True)
df['garage_area'].fillna(0, inplace=True)

# under the assumption that null is NA from the dictionary
df['pool_qc'].fillna('no_pool', inplace=True)

# under the assumption that null is NA from the dictionary
df['fence'].fillna('no_fence', inplace=True)

# under the assumption that null is NA from the dictionary
df['misc_feature'].fillna('no_misc_feature', inplace=True)

outliers = list(df[(df['saleprice'] < 200_000) & (df['gr_liv_area']>4000)].index)
df.drop(outliers, inplace=True)

# need to drop id and pid data as they to not influence that saleprice value
df.drop(columns=['id', 'pid'], inplace=True)
df.to_csv('../data/cleaned_data.csv', index=False)