import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn
import sklearn.preprocessing
from scipy.stats import skew, kurtosis

dataset = pd.read_csv('netflix_titles.csv', encoding='ISO-8859-1')
print(dataset.head())
print(dataset.info())

columns = dataset.columns

for i in columns:
    print(f'Number of null values in {i}: {dataset[i].isnull().sum()}')

dataset[['director', 'cast', 'country']] = dataset[['director', 'cast', 'country']].replace(np.nan, 'Unknown')
dataset.dropna(subset=['rating', 'duration', 'date_added'], inplace=True)

print(dataset.describe(include=['object']))
print(f'Mean of Release Year: {dataset["release_year"].mean()}')
print(f'Median of Release Year: {dataset["release_year"].median()}')
print(f'Mode of Release Year: {dataset["release_year"].mode()}')

print(f'Dataset shape: {dataset.shape}')
print(f'Number of duplicated values: {dataset.duplicated().sum()}')

dataset['date_added'] = dataset['date_added'].apply(pd.to_datetime)
print(f'Data type of date_added: {dataset["date_added"].dtype}')

df_corr = dataset[['release_year', 'duration']].dropna()
print("\nSkewness and Kurtosis:")
for column in ['release_year', 'duration_mins']:
    if column in df_corr.columns:
        print(f"{column}: Skewness={skew(df_corr[column]):.2f}, Kurtosis={kurtosis(df_corr[column]):.2f}")
plt.figure(figsize=(6, 4))
c1 = dataset['type'].value_counts()

print(c1)
c1.plot(kind='bar')
plt.xlabel('Type')
plt.ylabel('Frequency')
plt.title('Count of Movies vs TV Shows')
plt.show()

def remove_outlier(col):
    sorted(col)
    Q1,Q3=np.percentile(col,[25,75])
    IQR=Q3-Q1
    lower_range= Q1-(1.5 * IQR)
    upper_range= Q3+(1.5 * IQR)
    return lower_range, upper_range

le = sklearn.preprocessing.LabelEncoder()
dataset['rating'] = le.fit_transform(dataset['rating'])

lr,ur=remove_outlier(dataset['rating'])
dataset['rating']=np.where(dataset['rating']>ur,ur,dataset['rating'])
dataset['rating']=np.where(dataset['rating']<lr,lr,dataset['rating'])

rating_counts = dataset['rating'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(rating_counts, labels=rating_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Set3', n_colors=len(rating_counts)))
plt.title('Distribution of Movies by Age Rating')
plt.axis('equal')  # Equal aspect ratio ensures that pie chart is circular.
plt.show()

dataset['duration'] = dataset['duration'].apply(lambda x: int(str(x).replace(' min', '')) if 'min' in str(x) else 0)
movies = dataset[dataset['type'] == 'Movie']
plt.figure(figsize=(10, 6))
sns.histplot(movies['duration'], kde=True, bins=30)
plt.title('Distribution of Movie Durations')
plt.xlabel('Duration (minutes)')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 6))
top_genres = dataset['listed_in'].value_counts().head(10)
sns.barplot(x=top_genres.values, y=top_genres.index, palette='viridis')
plt.title('Top 10 Genres on Netflix')
plt.xlabel('Number of Titles')
plt.ylabel('Genres')
plt.show()

plt.figure(figsize=(12, 8))

# Selecting two columns for correlation (e.g., release_year and duration)
correlation = dataset[['release_year', 'duration']].corr()

# Plotting the heatmap for the selected columns
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
plt.title('Correlation Heatmap Between Release Year and Duration')
plt.show()


plt.figure(figsize=(12, 6))
top_countries = dataset['country'].value_counts().head(10)
sns.barplot(x=top_countries.index, y=top_countries.values, palette='Set2')
plt.title('Top 10 Countries by Content Count')
plt.ylabel('Number of Titles')
plt.xticks(rotation=45)
plt.show()

dataset['year_added'] = dataset['date_added'].dt.year
plt.figure(figsize=(10, 6))
sns.histplot(dataset['year_added'].dropna(), kde=True, bins=15)
plt.title('Content Added by Year')
plt.xlabel('Year')
plt.ylabel('Count')
plt.show()
