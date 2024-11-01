import requests
import pandas as pd
import time


# Функція для отримання детальної інформації про книгу
def fetch_book_details(key):
    url = f"https://openlibrary.org{key}.json"
    response = requests.get(url)
    return response.json()


# Функція для отримання даних про книги з Open Library
def fetch_books(query, max_results=100):
    url = f"https://openlibrary.org/search.json?q={query}&limit={max_results}"
    response = requests.get(url)
    data = response.json()
    books = []
    counter = 0
    for doc in data.get('docs', []):
        book = {
            'title': doc.get('title', 'N/A'),
            'author': ', '.join(doc.get('author_name', ['N/A'])),
            'publish_year': doc.get('first_publish_year', 'N/A'),
            'isbn': ', '.join(doc.get('isbn', ['N/A'])),
            'key': doc.get('key', 'N/A'),
            'genre': query
        }
        # Отримуємо деталі книги
        details = fetch_book_details(book['key'])
        if isinstance(details.get('description'), dict):
            book['description'] = details['description'].get('value', 'N/A')
        else:
            book['description'] = details.get('description', 'N/A')  # Якщо опис рядок, просто беремо його
        books.append(book)
        counter += 1
        print(f'Get {counter}/100 books')
        time.sleep(1)  # Додаємо затримку для уникнення перевантаження API

    return books


# Розширений список запитів для отримання книг
search_queries = [
    'fiction', 'science', 'history', 'fantasy', 'mystery',
    'romance', 'biography', 'self-help', 'cookbook', 'travel',
    'children', 'young adult', 'horror', 'philosophy', 'poetry',
    'classic literature', 'graphic novels', 'science fiction', 'adventure', 'true crime'
]

all_books = []

for query in search_queries:
    print(f'Processing {query}')
    books = fetch_books(query)
    all_books.extend(books)

# Створення DataFrame
books_df = pd.DataFrame(all_books)

# Збереження в CSV файл
books_df.to_csv('books_database_with_descriptions.csv', index=False, encoding='utf-8')

print("База знань успішно створена і збережена у файлі 'books_database_with_descriptions.csv'.")
