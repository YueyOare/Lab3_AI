import time

import pandas as pd
from fuzzywuzzy import fuzz
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from main import get_api_key

# Створення клієнта для роботи з API моделі
client = InferenceClient(api_key=get_api_key())

# Завантаження бази даних книг і жанрів
books_df = pd.read_csv('books_features.csv')
genre_mapping = pd.read_csv('genre_mapping.csv')
genres_list = [
    'fiction', 'science', 'history', 'fantasy', 'mystery',
    'romance', 'biography', 'self-help', 'cookbook', 'travel',
    'children', 'young adult', 'horror', 'philosophy', 'poetry',
    'classic literature', 'graphic novels', 'science fiction', 'adventure', 'true crime'
]

# Функція для аналізу запиту користувача за допомогою Mistral
def analyze_user_query(query, max_retries=50, retry_delay=5):
    genres_str = ", ".join(genres_list)
    messages = [
        {
            "role": "user",
            "content": f"Extract from this text:\n1. Book title (if any, otherwise 'Not provided')\n2. Author name (if any, otherwise 'Not provided')\n3. Possible genres (only from the following list: {genres_str})\n4. Main keywords for book description\n\nText: {query}\n\nPlease respond only in English in any case, follow the specified format, and do not generate book titles or author names if they are not explicitly mentioned in the text. The response format should be:\n1. Book title: <title>\n2. Author name: <author>\n3. Genres: <genres>\n4. Keywords: <keywords>"
        }
    ]

    # Спроба зробити запит до моделі з обмеженою кількістю спроб
    for attempt in range(max_retries):
        stream = client.chat.completions.create(
            model="mistralai/Mistral-Nemo-Instruct-2407",
            messages=messages,
            max_tokens=500,
            stream=True
        )

        result_text = ""
        for chunk in stream:
            if hasattr(chunk, 'error') and chunk.error:
                print(f"Error: {chunk.error}")
                time.sleep(retry_delay)
                continue

            result_text += chunk.choices[0].delta.content

        if not result_text.strip():
            time.sleep(retry_delay)
            continue

        # Обробка відповіді моделі
        result_text = result_text.lower()
        result_text_splitted = result_text.split('\n')

        title = result_text_splitted[0].split(": ")[1] if "book title" in result_text_splitted[0] else None
        title = title if title != "not provided" else None
        author = result_text_splitted[1].split(": ")[1] if "author name" in result_text_splitted[1] else None
        author = author if author != "not provided" else None
        genres = [genre.strip() for genre in result_text_splitted[2].split(": ")[1].split(", ") if genre.strip() in genres_list] if result_text_splitted[2] else genres_list
        keywords = result_text_splitted[3].split(": ")[1] if "keywords" in result_text_splitted[3] else result_text

        return title, author, genres, keywords

    print("Max retries exceeded. Unable to complete the operation.")
    return None, None, None, None

# Завантаження моделі для векторизації опису книг
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Функція для пошуку книг за отриманими характеристиками
def search_books(title, author, genres, keywords, books_df):
    filtered_books = books_df

    results = {}

    # Vectorize the query
    query_combined_text = "Title: " + title if title is not None else ""
    query_combined_text += ". Author: " + author if author is not None else ""
    query_combined_text += ". Keywords: " + keywords
    query_vector = embedding_model.encode([query_combined_text], convert_to_numpy=True)

    # Calculate similarity for each book
    for index, book in filtered_books.iterrows():
        book_title = str(book['title'])
        book_author = book['author']

        try:
            title_similarity = fuzz.token_set_ratio(title.lower(), book_title.lower()) if title is not None else 0
        except Exception:
            title_similarity = 0
        try:
            author_similarity = fuzz.token_set_ratio(author.lower(), book_author.lower())
        except Exception:
            author_similarity = 0

        book_vector = book[[f'feature_{i}' for i in range(query_vector.shape[1])]].values.reshape(1, -1)
        description_similarity = cosine_similarity(query_vector, book_vector)[0][0]

        # Check if the book is already in the results
        book_key = (book_title, book_author)
        if book_key in results:
            # If the book is already in the results, add the genre to the existing entry
            results[book_key]['genre'].append(book['genre'])
        else:
            # If the book is not in the results, add a new entry
            results[book_key] = {
                'title': book_title,
                'author': book_author,
                'genre': [book['genre']],
                'title_similarity': title_similarity,
                'author_similarity': author_similarity,
                'description_similarity': description_similarity
            }

    # Convert the results dictionary to a list of dictionaries
    results = list(results.values())

    # Sort the results by similarity
    results = sorted(results,
                     key=lambda x: (x['title_similarity'], x['author_similarity'], x['description_similarity']),
                     reverse=True)

    return results


# Запит користувача
user_query = input("Опишіть, яку книгу ви шукаєте: ")

# Обробка запиту з допомогою Mistral
title, author, genres, keywords = analyze_user_query(user_query)

print(f"Запит: {user_query}")
print(f"Назва книги: {title}")
print(f"Автор: {author}")
print(f"Жанри: {', '.join(genres)}")
print(f"Ключові слова: {keywords}")

# Перевірка, чи вдалося обробити запит
if title is None and author is None and genres is None and keywords is None:
    print("Не вдалося обробити запит. Програма завершує роботу.")
else:
    # Пошук книг
    recommendations = search_books(title, author, genres, keywords, books_df)
    print("Рекомендовані книги:")
    for i, rec in enumerate(recommendations[:10]):
        print(f"{i+1}. Назва: {rec['title']}")
        print(f"   Автор: {rec['author']}")
        print(f"   Жанр: {rec['genre']}")
        print(f"   Схожість за назвою: {rec['title_similarity']}")
        print(f"   Схожість за автором: {rec['author_similarity']}")
        print(f"   Схожість за описом: {rec['description_similarity']}")
        print()
