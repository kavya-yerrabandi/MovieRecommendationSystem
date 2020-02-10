import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

credits_data = pd.read_csv('tmdb-movie-metadata/tmdb_5000_credits.csv')
movie_data = pd.read_csv('tmdb-movie-metadata/tmdb_5000_movies.csv')
credits_data.columns = ['id', 'title', 'cast', 'crew']
movie_data = movie_data.merge(credits_data, on='id')

popular = movie_data.sort_values('popularity', ascending=False)
plt.figure(figsize=(12, 4))

ax = sns.barplot(x=popular['popularity'].head(10), y=popular['original_title'].head(10), data=popular, palette="rocket")
plt.xlabel("Popularity")
plt.ylabel("Movie Title")
plt.title("Popular Movies")
fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig('results/popular_movies.png')

# Content Based Filtering
movie_data['overview'] = movie_data['overview'].fillna('')

# Remove the words such as "a", "an", "the"
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movie_data['overview'])
print(tfidf_matrix.shape)

cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movie_data.index, index=movie_data['original_title']).drop_duplicates()


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
    """ Gets the recommended movies for each title in the list
    :param movie_titles: List of movies
    :return: Recommended movie list
    """
    final_movie_list = []
    for movie_title in movie_titles:
        final_movie_list.extend(get_content_based_movie_recommendations(movie_title))
    return final_movie_list

final_movie_list = get_multiple_movie_recommendations_list(['Spy Kids', 'Avatar'])
print(get_multiple_movie_recommendations_list(['Spy Kids', 'Avatar']))
recommended_movie_data_file = open("results/recommended_movie_data_file.html", "w")

report_string = ''
report_string += "<html>\n<header>\n    <title>Movie list</title>\n</header>\n<body>"
report_string += "    <table border=\"2\">\n"
report_string += "        <tr>\n"
report_string += "            <td>No.</td>\n"
report_string += "            <td>Movie Title</td>\n"
report_string += "        </tr>\n"
for movie in final_movie_list:
    report_string += "        <tr>\n"
    report_string += "            <td>" + str(final_movie_list.index(movie)+1) + "</td>\n"
    report_string += "            <td>" + movie + "</td>\n"
    report_string += "        </tr>\n"
report_string += "    </table>\n</body>\n</html>\n\n\n"
recommended_movie_data_file.write(report_string)
recommended_movie_data_file.close()

