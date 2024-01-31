import numpy as np
import pandas as pd

# ᗜˬᗜ - subiect de test 2023
'''
The Overnights.csv file contains information on the number of overnight stays in tourist reception structures at locality level according to NSI (National Statistics Institute). The information is as follows:
• Siruta - Siruta Code of the locality;
• Years - The year in which the report was made;
• Value - Number of nights.
In the file RomaniaCodes.xlsx are the codifications of the localities, counties, regions and macro-regions of Romania according to NUTS (Nomenclature of Territorial Units for Statistics in Romania).
'''

rawOvernights = pd.read_csv('./dataIN/Overnights.csv', index_col=0)
rawPop = pd.read_csv('./dataIN/LocalitiesPopulation.csv', index_col=0)
rawCodes = pd.read_excel('./dataIN/RomaniaCodes.xlsx', sheet_name=[0, 1], index_col=0)

merged = rawOvernights \
.merge(right=rawPop, left_index=True, right_index=True) \
.merge(right=rawCodes[0], left_index=True, right_index=True) \
.merge(right=rawCodes[1], left_on='County', right_index=True)

# req 1
'''
Save into the Overnights_2016.csv file the values regarding the overnight stays at the level of 2016 at the locality level. The Siruta code, the name of the locality and the number of overnight stays will be saved for each locality.
'''

merged[merged['Ani'] == 2016][['City', 'Valoare']] \
.to_csv('./dataOUT/Overnights_2016.csv')

# req 2
'''
Save into the file CountyOvernights_2015.csv the values regarding the overnight stays by counties at the level of 2015.
'''

merged[merged['Ani'] == 2015][['NumeJudet', 'Valoare']] \
.groupby('NumeJudet').sum() \
.to_csv('./dataOUT/CountyOvernights_2015.csv')

# req 3
'''
Save into the file YearCountyOvernights.csv the average number of overnight stays by counties and years. The weighted average with the population of the localities will be calculated.
'''

merged[['Valoare', 'Ani', 'NumeJudet', 'Populatie']] \
.groupby(['Ani', 'NumeJudet']) \
.apply(func=lambda row: np.average(row['Valoare'], weights=row['Populatie'])) \
.to_csv('./dataOUT/YearCountryOvernights.csv')

# req 4
'''
Save into the YearOvernights.csv file the values regarding the overnight stays at locality level, by years. The values will be placed on separate columns, one column for each year.
'''

merged[['Ani', 'Valoare']] \
.pivot(columns="Ani", values="Valoare") \
.fillna(0) \
.to_csv('./dataOUT/YearOvernights.csv')

# req 5
'''
Save into the file CountyOvernights.csv the values regarding the overnight stays at county level, by years. The values will be placed in separate columns, one column for each year.
'''

merged[['Ani', 'Valoare', 'NumeJudet']] \
.groupby(['NumeJudet', 'Ani']).sum() \
.reset_index(1) \
.pivot(columns="Ani", values="Valoare") \
.to_csv('./dataOUT/CountyOvernights.csv')