########################
#RatingProducts
####################### #

# - Average
# - Time-Based Weighted Average
# - User-Based Weighted Average
# - Weighted Rating


####################
# Application: User and Time Weighted Course Score Calculation
####################
import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# (50+ Hours) Python A-Z™: Data Science and Machine Learning
# Score: 4.8 (4.764925)
# Total Points: 4611
# Score Percentages: 75, 20, 4, 1, <1
# Approximate Numerical Equivalents: 3458, 922, 184, 46, 6

df = pd.read_csv("Measurement Problems/datasets/course_reviews.csv")
df.head()
df.shape

# rating distribution
df["Rating"].value_counts()
df["Questions Asked"].value_counts()

# Score given based on questions asked
df.groupby("Questions Asked").agg({"Questions Asked": "count",
                                    "rating": "mean"})
df.head()

#######################
#Average
#######################

# Let's get their Average Scores
df["Rating"].mean()

#######################
# Time-Based Weighted Average
#######################
# Weighted Average by Point Times
# (give different weights to 30 days and different weights to 60 days, etc.)

df.head()
df.info()

# To make a calculation based on time, it is necessary to convert it to a time variable
df["Timestamp"]= pd.to_datetime(df["Timestamp"])

# We need to set a date for an old data set
current_date= pd.to_datetime("2021-02-10 0:0:0")

# Let's subtract the date of all comments from today's date and express it in days
df["days"]=(current_date-df["Timestamp"]).dt.days

# To access comment RATINGS for the last 30 days
df.loc[df["days"] <= 30, "Rating"].count()

df.loc[df["days"] <= 30, "Rating"].mean()

# Comments from more than 30 days and less than 90 days are for RATINGS
df.loc[(df["days"] > 30) & (df["days"] <=90), "Rating"].mean()

# Ratings greater than 90, less than 180 and equal
df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean()

# Ratings greater than 180 days
df.loc[(df["days"] > 180), "Rating"].mean()

# We can reflect the effect of time on the weight calculation by giving different weights to focus on different time intervals.
df.loc[df["days"] <= 30, "Rating"].mean() * 28/100 + \
     df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean() * 26/100 + \
     df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean() * 24/100 + \
     df.loc[(df["days"] > 180), "Rating"].mean() * 22/100

# Functionalization

def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
     return dataframe.loc[dataframe["days"] <= 30, "Rating"].mean() * w1 / 100 + \
            dataframe.loc[(dataframe["days"] > 30) & (dataframe["days"] <= 90), "Rating"].mean() * w2 / 100 + \
            dataframe.loc[(dataframe["days"] > 90) & (dataframe["days"] <= 180), "Rating"].mean() * w3 / 100 + \
            dataframe.loc[(dataframe["days"] > 180), "Rating"].mean() * w4 / 100


time_based_weighted_average(df)

time_based_weighted_average(df, 30, 26, 22, 22)

#######################
# User-Based Weighted Average
#######################

# The points given by all users must have the same meaning (all those who watched the course and those who watched a certain part of it).
df.head()

df.groupby("Progress").agg({"Rating": "mean"})


df.loc[df["Progress"] <= 10, "Rating"].mean() * 22 / 100 + \
     df.loc[(df["Progress"] > 10) & (df["Progress"] <= 45), "Rating"].mean() * 24 / 100 + \
     df.loc[(df["Progress"] > 45) & (df["Progress"] <= 75), "Rating"].mean() * 26 / 100 + \
     df.loc[(df["Progress"] > 75), "Rating"].mean() * 28 / 100



def user_based_weighted_average(dataframe, w1=22, w2=24, w3=26, w4=28):
     return dataframe.loc[dataframe["Progress"] <= 10, "Rating"].mean() * w1 / 100 + \
            dataframe.loc[(dataframe["Progress"] > 10) & (dataframe["Progress"] <= 45), "Rating"].mean() * w2 / 100 + \
            dataframe.loc[(dataframe["Progress"] > 45) & (dataframe["Progress"] <= 75), "Rating"].mean() * w3 / 100 + \
            dataframe.loc[(dataframe["Progress"] > 75), "Rating"].mean() * w4 / 100


user_based_weighted_average(df, 20, 24, 26, 30)

#######################
# Weighted Rating
#######################

def course_weighted_rating(dataframe, time_w=50, user_w=50):
     return time_based_weighted_average(dataframe) * time_w/100 + user_based_weighted_average(dataframe)*user_w/100

course_weighted_rating(df)

course_weighted_rating(df, time_w=40, user_w=60)