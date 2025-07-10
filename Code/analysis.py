# analysis.py

# ------------------------
# 📊 Applied Stats Project
# ------------------------

# ✅ Import libraries
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
from scipy.stats import norm

# ✅ Set plot style
palette1 = ['#0f4662', '#7994a0', '#a9becb', '#dbe5ea']
sns.set(style="whitegrid")

# ✅ Load data
data = pd.read_csv('../data/data1.csv')  # Adjust path as needed
print(data.head())
print(data.describe())

# ✅ Convert non-numeric attribute "Type of music" into integer values
# You might have done this manually in the actual dataset already
music_mapping = {
    "1-Very Sad & Dark": 1,
    "2-Melancholic & Emotional": 2,
    "3-Neutral & Balanced": 3,
    "4-Upbeat & Cheerful": 4,
    "5-Extremely Happy & Euphoric": 5
}
data['musicHappyInt'] = data['musicHappy'].map(music_mapping)

# ✅ Histogram of CGPA
plt.figure(figsize=(10,5))
sns.histplot(data['CGPA'], kde=False, color=palette1[1], bins=15)
plt.title('CGPA Distribution')
plt.ylabel('Number of Students')
plt.xlabel('CGPA')
plt.tight_layout()
plt.show()

# ✅ Bar plots for other categorical columns
for column in data.columns:
    if column not in ['Timestamp', 'CGPA', 'UG/PG', 'Year', 'musicHappy', 'musicHappyInt']:
        attribute = data[column].value_counts().sort_index()
        plt.figure(figsize=(10, 5))
        sns.barplot(x=attribute.index, y=attribute.values, color=palette1[2])
        plt.yticks(range(0, int(attribute.max() + 2), 2))
        plt.ylabel('Number of Students')
        plt.xlabel(column)
        plt.title(f'{column} vs Frequency')
        plt.tight_layout()
        plt.show()

# ✅ Pie chart for UG/PG
graduation_types = data['UG/PG'].value_counts()
explode = [0.1] + [0 for _ in range(len(graduation_types) - 1)]
colors = ['#ffcc99', '#FF0000', '#99ff99', '#ff9999']
plt.figure(figsize=(5, 5))
plt.pie(graduation_types, autopct='%1.1f%%', explode=explode, colors=colors, startangle=140, shadow=True)
plt.title('UG vs PG Students')
plt.tight_layout()
plt.show()

# ✅ Pie chart for music preferences
song_types = data['musicHappy'].value_counts()
explode = [0.1] * len(song_types)
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']
plt.figure(figsize=(8, 8))
plt.pie(song_types, labels=song_types.index, autopct='%1.1f%%', explode=explode, startangle=90)
plt.title('Type of Songs People Listen To')
plt.tight_layout()
plt.show()

# ✅ Box plots for selected numeric attributes
need_boxPlot = ['CGPA', 'clubsNo', 'branchDifficulty', 'religion', 'musicHappyInt', 'happinessGeneral']
for column in need_boxPlot:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=data[column], color=palette1[1], orient='h', whis=(0, 100))
    plt.title(f'Box Plot for {column}')
    plt.ylabel(column)
    plt.tight_layout()
    plt.show()
    print(data[column].describe())

# ✅ Ogives (Cumulative Frequency)
attributes = ['clubsNo', 'CGPA', 'happinessGeneral']
for attribute in attributes:
    totalAttribute = data[attribute].value_counts().sort_index().cumsum()
    plt.figure(figsize=(10, 5))
    plt.plot(totalAttribute.index, totalAttribute.values, marker='o', color=palette1[0])
    plt.title(f'Ogive for {attribute}')
    plt.xlabel(attribute)
    plt.ylabel('Cumulative Frequency')
    plt.tight_layout()
    plt.show()

# ✅ Year-wise happiness based on UG/PG
yearwise_cgpa = data.groupby(['Year', 'UG/PG'])[['happinessGeneral']].agg(['mean', 'max', 'min', 'count'])
print(yearwise_cgpa)

# ✅ Correlation heatmap
numeric_data = data.select_dtypes(include=[np.number])
correlation_matrix = numeric_data.corr()
cols_of_interest = ['nightOwl','CGPA','supportFamily','posToMusic','happinessGeneral']
cgpa_correlation = correlation_matrix[cols_of_interest]

plt.figure(figsize=(10, 8))
sns.heatmap(cgpa_correlation, annot=True, cmap='coolwarm', linewidths=1)
plt.title('Correlation Heat Map')
plt.tight_layout()
plt.show()

# ✅ Group happiness by Year
grouped = data.groupby(['Year'])['lifeSatisfaction'].agg(['mean', 'max', 'min', 'count'])
print(grouped)

# ✅ Central Limit Theorem: Happiness
sample_size = 30
num_samples = 10000
sample_means = [data['happinessGeneral'].sample(sample_size, replace=True).mean() for _ in range(num_samples)]

plt.figure(figsize=(10, 5))
sns.histplot(sample_means, kde=False, stat="density", bins=30, color=palette1[1])
x = np.linspace(min(sample_means), max(sample_means), 100)
p = norm.pdf(x, np.mean(sample_means), np.std(sample_means))
plt.plot(x, p, 'k', linewidth=2, label='Normal Distribution')
plt.title('CLT: Sample Means of Happiness')
plt.xlabel('Sample Mean')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.show()

# ✅ Central Limit Theorem: Religion
sample_means = [data['religion'].sample(sample_size, replace=True).mean() for _ in range(num_samples)]

plt.figure(figsize=(10, 5))
sns.histplot(sample_means, kde=False, stat="density", bins=30, color=palette1[2])
x = np.linspace(min(sample_means), max(sample_means), 100)
p = norm.pdf(x, np.mean(sample_means), np.std(sample_means))
plt.plot(x, p, 'k', linewidth=2, label='Normal Distribution')
plt.title('CLT: Sample Means of Religion')
plt.xlabel('Sample Mean')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.show()
