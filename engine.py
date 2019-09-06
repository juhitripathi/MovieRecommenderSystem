import os
from pyspark.mllib.recommendation import ALS

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_counts_and_averages(id_and_ratings_tuple):
    """Count average ratings for the given ratings_RDD
    :params id_and_ratings_tuple: A tuple with movieId and a list of movie ratings
    :returns: tuple with counts and averages corresponding to movieId
    """
    num_ratings = len(id_and_ratings_tuple[1])
    return ID_and_ratings_tuple[0], (num_ratings, float(sum(x for x in id_and_ratings_tuple[1]))/num_ratings)


class RecommendationEngine

    def __init__(self, sc, dataset_path):
        """Initialize the recommendation engine given a Spark context and a dataset path
        :params sc: Spark context
        :params dataset_path: Path for the training dataset
        """
        self.sc = sc
        # Load ratings data for later use
        logger.info("Loading Ratings data...")
        ratings_file_path = os.path.join(dataset_path, 'ratings.csv')
        ratings_raw_RDD = self.sc.textFile(ratings_file_path)
        ratings_raw_data_header = ratings_raw_RDD.take(1)[0]
        self.ratings_RDD = ratings_raw_RDD.filter(lambda line: line!=ratings_raw_data_header)\
            .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),int(tokens[1]),float(tokens[2]))).cache()

        # Load movies data for later use
        logger.info("Loading Movies data...")
        movies_file_path = os.path.join(dataset_path, 'movies.csv')
        movies_raw_RDD = self.sc.textFile(movies_file_path)
        movies_raw_data_header = movies_raw_RDD.take(1)[0]
        self.movies_RDD = movies_raw_RDD.filter(lambda line: line!=movies_raw_data_header)\
            .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),tokens[1],tokens[2])).cache()
        self.movies_titles_RDD = self.movies_RDD.map(lambda x: (int(x[0]),x[1])).cache()

        # Pre-calculate movies ratings counts
        self.__count_and_average_ratings()

        # Train the model
        self.rank = 8
        self.seed = 5L
        self.iterations = 10
        self.regularization_parameter = 0.1
        self.__train_model()


    def __count_and_average_ratings(self):
        """Update movies ratings counts from current data self.ratings_RDD
        """
        logger.info("Counting movie ratings...")
        movie_ID_with_ratings_RDD = self.ratings_RDD.map(lambda x: (x[1], x[2])).groupByKey()
        movie_ID_with_avg_ratings_RDD = movie_ID_with_ratings_RDD.map(get_counts_and_averages)
        self.movies_rating_counts_RDD = movie_ID_with_avg_ratings_RDD.map(lambda x: (x[0], x[1][0]))


    def __train_model(self):
        """Train a model with Alternating Least Square method
        """
        logger.info("Training the ALS model...")
        self.model = ALS.train(self.ratings_RDD, self.rank, seed=self.seed,
                               iterations=self.iterations, lambda_=self.regularization_parameter)
        logger.info("ALS model built!")


    def __predict_movie_ratings(self, user_movie_RDD):
        """Gets predictions for a given (userID, movieID) formatted RDD
        :params user_movie_rdd: User-Movie RDD
        :return: RDD having [movieTitle, movieRating, numRatings]
        """
        predicted_RDD = self.model.predictAll(user_and_movie_RDD)
        predicted_rating_RDD = predicted_RDD.map(lambda x: (x.product, x.rating))
        predicted_rating_title_and_count_RDD = \
            predicted_rating_RDD.join(self.movies_titles_RDD).join(self.movies_rating_counts_RDD)
        predicted_rating_title_and_count_RDD = \
            predicted_rating_title_and_count_RDD.map(lambda r: (r[1][0][1], r[1][0][0], r[1][1]))

        return predicted_rating_title_and_count_RDD

    def add_ratings(self, ratings):
        """Add new user-rated movies to the list and retrain the model
        :params ratings: List of rating tuples with each tuple in format (user_id, movie_id, rating)
        """
        new_ratings_RDD = self.sc.parallelize(ratings)
        self.ratings_RDD = self.ratings_RDD.union(new_ratings_RDD)
        self.__count_and_average_ratings()
        self.__train_model()

    def get_ratings_for_movie_ids(self, user_id, movie_ids):
        """Given a user_id and a list of movie_ids, predict their ratings
        :params user_id: id corresponding to the user
        :params movie_ids: List of movie ids
        :returns: returns ratings corresponding to the provided movie_ids
        """
        requested_movies_RDD = self.sc.parallelize(movie_ids).map(lambda x: (user_id, x))
        ratings = self.__predict_movie_ratings(requested_movies_RDD).collect()
        return ratings

    def top_rated_movies(self, user_id, count):
        """Function for recommending top unrated movies by the user.
        :param user_id: Id corresponding to the user
        :param count: Top count movies to recommend
        :return: returns top count number of recommended movies
        """
        user_unrated_movies_RDD = self.ratings_RDD.filter(lambda rating: not rating[0] == user_id)\
                                                 .map(lambda x: (user_id, x[1])).distinct()
        ratings = self.__predict_movie_ratings(user_unrated_movies_RDD).filter(lambda r: r[2]>=25).takeOrdered(count, key=lambda x: -x[1])
        return ratings
