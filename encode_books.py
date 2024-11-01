import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Завантаження даних
books_df = pd.read_csv('books_database_with_descriptions.csv')

# Ініціалізація моделі для ембеддінгів
model = SentenceTransformer('all-MiniLM-L6-v2')

# Об'єднуємо автора, назву й опис в один текстовий блок
books_df['combined_text'] = (
    "Title: " + books_df['title'].fillna('') +
    ". Author: " + books_df['author'].fillna('') +
    ". Description: " + books_df['description'].fillna('')
)

# Векторизація об'єднаного тексту з використанням SentenceTransformer
description_vectors = model.encode(books_df['combined_text'].tolist(), convert_to_numpy=True)

# Масштабування року публікації
scaler = StandardScaler()
books_df['publish_year_scaled'] = scaler.fit_transform(books_df[['publish_year']].fillna(0))

# Бінарне кодування жанру
# genre_encoded = pd.get_dummies(books_df['genre'], prefix='genre')
#
# # Зберігаємо маппінг жанрів у окремий файл
# genre_mapping = pd.DataFrame({
#     'genre': genre_encoded.columns,
#     'encoding_index': range(len(genre_encoded.columns))
# })
# genre_mapping.to_csv('genre_mapping.csv', index=False, encoding='utf-8')
# print("Маппінг жанрів збережено у файлі 'genre_mapping.csv'")

# Збирання всіх векторів ознак в один вектор для кожної книги
features_matrix = np.hstack([
    description_vectors,
    books_df[['publish_year_scaled']].values,
])

# Створення DataFrame та збереження у файл CSV
features_df = pd.DataFrame(features_matrix)
features_df.columns = [f'feature_{i}' for i in range(features_df.shape[1])]
features_df['title'] = books_df['title']
features_df['author'] = books_df['author']
features_df['genre'] = books_df['genre']

# Зберігаємо у файл CSV
features_df.to_csv('books_features.csv', index=False, encoding='utf-8')
print("Вектори ознак успішно збережено у файлі 'books_features.csv'")
