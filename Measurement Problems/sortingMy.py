####################### #
#SortingProducts
####################### #

####################### #
# Application: Course Sorting
####################### #
import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("Measurement Problems/datasets/product_sorting.csv")
print(df.shape)
df.head(10)

#######################
# Sorting by Rating
#######################

# Let's rank them according to the rating, with the score at the top.

df.sort_values("rating", ascending=False).head(20)

#######################
# Sorting by Comment Count or Purchase Count
#######################
# Let's sort by number of purchases
df.sort_values("purchase_count", ascending=False).head(20)

# Let's sort by the number of comments
df.sort_values("commment_count", ascending=False).head(20)

#######################
# Sorting by Rating, Comment and Purchase
#######################

# Keeping three factors together at the same time. Catching them accordingly.
# Let's get closer by bringing them all to the same scale

df["purchase_count_scaled"] = MinMaxScaler(feature_range=(1, 5)). \
     fit(df[["purchase_count"]]).transform(df[["purchase_count"]])

df.describe().T

# We should do a similar process for comment_count

df["comment_count_scaled"] = MinMaxScaler(feature_range=(1, 5)). \
     fit(df[["commment_count"]]).transform(df[["commment_count"]])

df.head()

# If these variables are of the same type, we can take their average.
# We can weight it and thus get a score (formed by weighting)

(df["comment_count_scaled"] * 32 / 100 + df["purchase_count_scaled"] * 26 / 100 + df["rating"] * 42 / 100)

# Let's functionalize
def weighted_sorting_score(dataframe, w1=32, w2=26, w3=42):
     return (dataframe["comment_count_scaled"] * w1 / 100 +
             dataframe["purchase_count_scaled"] * w2 / 100 +
             dataframe["rating"] * w3 / 100)


df["weighted_sorting_score"] = weighted_sorting_score(df)

df.sort_values("weighted_sorting_score", ascending=False).head(20)

df[df["course_name"].str.contains("Data Science")].sort_values("weighted_sorting_score", ascending=False).head(20)

#######################
# Bayesian Average Rating Score
#######################

# Sorting Products with 5 Star Rated
# Sorting Products According to Distribution of 5 Star Rating

# Let's calculate a weighted probabilistic average using the distribution information of the scores

def bayesian_average_rating(n, confidence=0.95):
     if sum(n) == 0:
         return 0
     K = len(n)
     z = st.norm.ppf(1 - (1 - confidence) / 2)
     N = sum(n)
     first_part = 0.0
     second_part = 0.0
     for k, n_k in enumerate(n):
         first_part += (k + 1) * (n[k] + 1) / (N + K)
         second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
     score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
     return score


df.head()

# We add a new variable.
df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["1_point",
                                                                 "2_point",
                                                                 "3_point",
                                                                 "4_point",
                                                                 "5_point"]]), axis=1)

# Let's sort it
df.sort_values("weighted_sorting_score", ascending=False).head(20)

# Let's sort according to the bar_score we just created.
# bar_score gives us a ranking by focusing only on ratings.
df.sort_values("bar_score", ascending=False).head(20)

# Let's bring the course names according to the index
df[df["course_name"].index.isin([5, 1])].sort_values("bar_score", ascending=False)

#######################
# Hybrid Sorting: BAR Score + Other Factors
#######################

#RatingProducts
# - Average
# - Time-Based Weighted Average
# - User-Based Weighted Average
# - Weighted Rating
# - Bayesian Average Rating Score

#SortingProducts
# - Sorting by Rating
# - Sorting by Comment Count or Purchase Count
# - Sorting by Rating, Comment and Purchase
# - Sorting by Bayesian Average Rating Score (Sorting Products with 5 Star Rated)
# - Hybrid Sorting: BAR Score + Other Factors
# We functionalize

def hybrid_sorting_score(dataframe, bar_w=60, wss_w=40):
     bar_score = dataframe.apply(lambda x: bayesian_average_rating(x[["1_point",
                                                                      "2_point",
                                                                      "3_point",
                                                                      "4_point",
                                                                      "5_point"]]), axis=1)
     wss_score = weighted_sorting_score(dataframe)

     return bar_score*bar_w/100 + wss_score*wss_w/100

df["hybrid_sorting_score"] = hybrid_sorting_score(df)

df.sort_values("hybrid_sorting_score", ascending=False).head(20)

# Upcoming courses in searches with data science keyword

df[df["course_name"].str.contains("Data Science")].sort_values("hybrid_sorting_score", ascending=False).head(20)


####################
# Application: IMDB Movie Scoring & Sorting
####################

import pandas as pd
import math
import scipy.stats as st
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("Measurement Problems/datasets/movies_metadata.csv",
                  low_memory=False) # To turn off DtypeWarning

df = df[["title", "vote_average", "vote_count"]]

df.head()
df.shape

#############################
# Sorting by Vote Average
#############################

# When we rank the movies according to their average scores
df.sort_values("vote_average", ascending=False).head(20)

# Information about the quarter values of the vote numbers after sorting them from smallest to largest.
# (We control where we can filter)
df["vote_count"].describe([0.10, 0.25, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99]).T

# As a condition, we listed those with a number of votes over 400.
df[df["vote_count"] > 400].sort_values("vote_average", ascending=False).head(20)

from sklearn.preprocessing import MinMaxScaler

# Let's use it together as we see its effects.
# Let's standardize the vote_count variable. Let's score it.
df["vote_count_score"] = MinMaxScaler(feature_range=(1, 10)).fit(df[["vote_count"]]). \
                          transform(df[["vote_count"]])

# Let's derive an average_count_score variable.
df["average_count_score"] = df["vote_average"] * df["vote_count_score"]

# Let's sort by this variable
df.sort_values("average_count_score", ascending=False).head(20)

#############################
# IMDB Weighted Rating
#############################


# weighted_rating = (v/(v+M) * r) + (M/(v+M) * C)

# r = vote average
# v = vote count
# M = minimum votes required to be listed in the Top 250
# C = the mean vote across the whole report (currently 7.0)

#Movie 1:
# r = 8
# M = 500
# v = 1000

# (1000 / (1000+500))*8 = 5.33


# Movie 2:
# r = 8
# M = 500
# v = 3000

# (3000 / (3000+500))*8 = 6.85

# (1000 / (1000+500))*9.5

#Movie 1:
# r = 8
# M = 500
# v = 1000

# First part:
# (1000 / (1000+500))*8 = 5.33

# Second part:
# 500/(1000+500) * 7 = 2.33

# Total = 5.33 + 2.33 = 7.66


# Movie 2:
# r = 8
# M = 500
# v = 3000

# First part:
# (3000 / (3000+500))*8 = 6.85

# Second part:
# 500/(3000+500) * 7 = 1

# Total = 7.85

M = 2500
C = df['vote_average'].mean()

def weighted_rating(r, v, M, C):
     return (v / (v + M) * r) + (M / (v + M) * C)

# Let's bring up our previous list.
df.sort_values("average_count_score", ascending=False).head(10)

# Let's take a look at our selections
weighted_rating(7.40000, 11444.00000, M, C)

weighted_rating(8.10000, 14075.00000, M, C)

weighted_rating(8.50000, 8358.00000, M, C)

# To apply to all data
df["weighted_rating"] = weighted_rating(df["vote_average"],
                                         df["vote_count"], M, C)

df.sort_values("weighted_rating", ascending=False).head(10)


#######################
# Bayesian Average Rating Score
#######################

#12481 The Dark Knight
#314 The Shawshank Redemption
#2843 Fight Club
#15480 Inception
#292 Pulp Fiction

# as function

def bayesian_average_rating(n, confidence=0.95):
     if sum(n) == 0:
         return 0
     K = len(n)
     z = st.norm.ppf(1 - (1 - confidence) / 2)
     N = sum(n)
     first_part = 0.0
     second_part = 0.0
     for k, n_k in enumerate(n):
         first_part += (k + 1) * (n[k] + 1) / (N + K)
         second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
     score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
     return score

bayesian_average_rating([34733, 4355, 4704, 6561, 13515, 26183, 87368, 273082, 600260, 1295351])

bayesian_average_rating([37128, 5879, 6268, 8419, 16603, 30016, 78538, 199430, 402518, 837905])


# Let's get the distribution of votes
df = pd.read_csv("Measurement Problems/datasets/imdb_ratings.csv")
df = df.iloc[0:, 1:]

# There are 250 movies, let's look at this ranking.
df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["one", "two", "three", "four", "five",
                                                                 "six", "seven", "eight", "nine", "ten"]]), axis=1)
df.sort_values("bar_score", ascending=False).head(20)

# Weighted Average Ratings
# IMDb publishes weighted vote averages rather than raw data averages.
# The simplest way to explain it is that although we accept and consider all votes received by users,
# not all votes have the same impact (or ‘weight’) on the final rating.

# When unusual voting activity is detected,
# an alternate weighting calculation may be applied in order to preserve the reliability of our system.
# To ensure that our rating mechanism remains effective,
# we do not disclose the exact method used to generate the rating.

# See also the complete FAQ for IMDb ratings.



