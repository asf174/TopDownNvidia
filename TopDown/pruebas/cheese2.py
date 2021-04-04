

import matplotlib.pyplot as plt
import numpy as np

#read the csv file
filename = '../LAB03-DATA VISUALIZATION USING MATPLOTLIB/singapore-residents-by-ethnic-group-and-sex-end-june-annual.csv'
data = np.genfromtxt(filename, dtype=['i8','U50','i8'], delimiter=',', names=True)

#extract datas that are of 1960
data_1960 = data[data['year'] == 1960]

#extract datas that are of 2016
data_2016 = data[data['year'] == 2016]

#extract datas that have this keyword 'TOTAL MALE RESIDENTS' and 'TOTAL FEMALE RESIDENTS'(1960)
male_and_female_1960 = data_1960[np.isin(data_1960['level_1'], ['Total Male Residents' , 'Total Female Residents'])]

#extract datas that have this keyword 'TOTAL MALE RESIDENTS' and 'TOTAL FEMALE RESIDENTS'(2016)
male_and_female_2016 = data_2016[np.isin(data_2016['level_1'], ['Total Male Residents' , 'Total Female Residents'])]

# PLOTTING OF THE 1960 PIE CHART-------------------------------------------------------------------------------------------------
labels = male_and_female_1960['level_1']
values = male_and_female_1960['value']

#settings and configs for the pie charts
colors = ['#FF8F33','#33FFDC']
explode = (0.1, 0)

plt.figure(figsize=(5,5))
plt.pie(values,labels = labels,colors = colors,autopct = '%1.1f%%')
plt.title('Gender Composition in 1960')

# PLOTTING OF THE 2016 PIE CHART-------------------------------------------------------------------------------------------------
labels = male_and_female_2016['level_1']
values = male_and_female_2016['value']

#settings and configs for the pie charts
colors = ['#FF8F33','#33FFDC']
explode = (0.1, 0)

plt.figure(figsize=(5,5))
plt.pie(values,labels = labels,colors = colors,autopct = '%1.1f%%')
plt.title('Gender Composition in 2016')

plt.show() 
ax.pie(sizes, labels=labels, autopct='%1.1f%%', explode=(0,0,0,0,0,0,0,0,0), startangle=0)
import matplotlib.pyplot as plt
import numpy as np

#read the csv file
filename = '../LAB03-DATA VISUALIZATION USING MATPLOTLIB/singapore-residents-by-ethnic-group-and-sex-end-june-annual.csv'
data = np.genfromtxt(filename, dtype=['i8','U50','i8'], delimiter=',', names=True)

#extract datas that are of 1960
data_1960 = data[data['year'] == 1960]

#extract datas that are of 2016
data_2016 = data[data['year'] == 2016]

#extract datas that have this keyword 'TOTAL MALE RESIDENTS' and 'TOTAL FEMALE RESIDENTS'(1960)
male_and_female_1960 = data_1960[np.isin(data_1960['level_1'], ['Total Male Residents' , 'Total Female Residents'])]

#extract datas that have this keyword 'TOTAL MALE RESIDENTS' and 'TOTAL FEMALE RESIDENTS'(2016)
male_and_female_2016 = data_2016[np.isin(data_2016['level_1'], ['Total Male Residents' , 'Total Female Residents'])]

# PLOTTING OF THE 1960 PIE CHART-------------------------------------------------------------------------------------------------
labels = male_and_female_1960['level_1']
values = male_and_female_1960['value']

#settings and configs for the pie charts
colors = ['#FF8F33','#33FFDC']
explode = (0.1, 0)

plt.figure(figsize=(5,5))
plt.pie(values,labels = labels,colors = colors,autopct = '%1.1f%%')
plt.title('Gender Composition in 1960')

# PLOTTING OF THE 2016 PIE CHART-------------------------------------------------------------------------------------------------
labels = male_and_female_2016['level_1']
values = male_and_female_2016['value']

#settings and configs for the pie charts
colors = ['#FF8F33','#33FFDC']
explode = (0.1, 0)

plt.figure(figsize=(5,5))
plt.pie(values,labels = labels,colors = colors,autopct = '%1.1f%%')
plt.title('Gender Composition in 2016')

plt.show() 
