import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

dataset = pd.read_csv('netflix_titles.csv')
print(dataset.head())
print(dataset.info())

columns = dataset.columns

for i in columns:
    print(f'Number of null values in {i}: {dataset[i].isnull().sum()}')

dataset[['director', 'cast', 'country']] = dataset[['director', 'cast', 'country']].replace(np.nan, 'Unknown')
dataset.dropna(subset=['rating', 'duration', 'date_added'], inplace=True)

print(dataset.describe(include=['object']))
print(f'Dataset shape: {dataset.shape}')
print(f'Number of duplicated values: {dataset.duplicated().sum()}')

dataset['date_added'] = dataset['date_added'].apply(pd.to_datetime)
print(f'Data type of date_added: {dataset["date_added"].dtype}')

plt.figure(figsize=(6, 4))
sns.countplot(data=dataset, x='type', palette='coolwarm')
plt.title('Count of Movies vs TV Shows')
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
correlation = dataset.corr(numeric_only=True)
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Encoded Features')
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