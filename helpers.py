def filter_movies(movies):
    filtered_movies = movies[movies.release_year != 'not found']
    filtered_movies = filtered_movies[(filtered_movies.release_year >= '1995') & (filtered_movies.language == 'en')]
    return filtered_movies