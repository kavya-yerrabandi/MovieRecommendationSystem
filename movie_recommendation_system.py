import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

credits_data = pd.read_csv('tmdb-movie-metadata/tmdb_5000_credits.csv')
movie_data = pd.read_csv('tmdb-movie-metadata/tmdb_5000_movies.csv')
credits_data.columns = ['id', 'title', 'cast', 'crew']
movie_data = movie_data.merge(credits_data, on='id')

movie_data_file = open("movie_data_file.html", "w")
movie_data_file.write(movie_data.head(5).to_html())
movie_data_file.close()

print(movie_data['overview'].head(5))

# Content Based Filtering

movie_data['overview'] = movie_data['overview'].fillna('')

# Remove the words such as "a", "an", "the"
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movie_data['overview'])
print(tfidf_matrix.shape)

cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movie_data.index, index=movie_data['original_title']).drop_duplicates()

C = movie_data['vote_average'].mean()
print(C)
m = movie_data['vote_count'].quantile(0.9)
print(m)
q_movies = movie_data.copy().loc[movie_data['vote_count'] >= m]
print(q_movies.shape)


def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)


q_movies['score'] = q_movies.apply(weighted_rating, axis=1)


# Sort movies based on score calculated above
q_movies = q_movies.sort_values('score', ascending=False)


print(q_movies[['original_title', 'vote_count', 'vote_average', 'score']].head(10))
top_movies_data_file = open("top_movies_data_file.html", "w")
# top_movies_data_file.close()

# pop = movie_data.sort_values('popularity', ascending=False)
popular = movie_data.sort_values('popularity', ascending=False)
plt.figure(figsize=(12, 4))

ax = sns.barplot(x=popular['popularity'].head(10), y=popular['original_title'].head(10), data=popular, palette="rocket")
# plt.barh(pop['title'].head(6), pop['popularity'].head(6), align='center',
#         color='skyblue')
# plt.gca().invert_yaxis()
plt.xlabel("Popularity")
plt.title("Popular Movies")
fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig('popular_movies.png')


def get_content_based_movie_recommendations(title, cosine_sim=cosine_sim_matrix):
    """ Takes the title of the movie and returns the list of recommended movies
    :param title: Movie title
    :param cosine_sim: Cosine similarity matrix
    :return: List of five similar movies
    """
    movie_index = indices[title]
    similarity_scores = list(enumerate(cosine_sim[movie_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:6]
    movie_indices = [i[0] for i in similarity_scores]
    return movie_data['original_title'].iloc[movie_indices]


def get_multiple_movie_recommendations_list(movie_titles):
    final_movie_list = []
    for movie_title in movie_titles:
        final_movie_list.extend(get_content_based_movie_recommendations(movie_title))
    return final_movie_list


print(get_multiple_movie_recommendations_list(['Spy Kids', 'Avatar']))


