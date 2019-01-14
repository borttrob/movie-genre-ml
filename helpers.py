import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd

def visualize_count_per_year_language(movies):
    movie_counts_per_year = movies[movies['release_year'] != 'not found'].release_year.value_counts().sort_index()
    movie_counts_per_year.index = movie_counts_per_year.index.astype('int')

    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    ax = movie_counts_per_year.plot()
    ax.set_xlabel('year')
    ax.set_ylabel('movie count')
    ax.set_title('Movie count per release year')

    plt.subplot(1, 2, 2)
    ax = movies.language.value_counts().sort_values(ascending=False)[:15].plot.bar(ax=plt.gca())
    ax.set_xlabel('language')
    ax.set_ylabel('movie count')
    ax.set_title('Movie count per language')


def calc_genre_counts(movie_genres):
    flat_list = [genre for movie_genres in movie_genres for genre in movie_genres]
    return calc_counts(flat_list, 'genre')

def calc_counts(series, col_name):
    counts = Counter(series)
    df = pd.DataFrame(list(counts.items()), columns=[col_name, 'count'])
    df = df.set_index(col_name)
    df = df.sort_values(['count'], ascending=False)
    df['distribution'] = df['count']/df['count'].sum()
    return df


def build_tags_column(movies, grouped_tags):
    tag_column = []
    for movieId in movies.index:
        try:
            movie_tags = grouped_tags.loc[movieId].index
            tag_column.append(list(movie_tags))
        except:
            tag_column.append([])
    return tag_column

def is_gpu_used():
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())

    gpu_devices = [device for device in device_lib.list_local_devices() if device ]

    return len(gpu_devices) > 0

def visualize_genre_count_distribution(movies):
    lengths_of_genres = movies['relevant_genres'].apply(lambda x: len(x))

    genres_count_average = np.mean(lengths_of_genres)
    print('Genre count average {}'.format(genres_count_average))
    genres_count_variance = np.var(lengths_of_genres)
    print('Genre count variance {}'.format(genres_count_variance))

    ax = lengths_of_genres.value_counts().plot(kind='bar')
    ax.set_xlabel('genre count')
    ax.set_ylabel('movies count')
    ax.set_title('Count of genres')