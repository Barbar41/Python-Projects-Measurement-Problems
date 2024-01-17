####################### #################
# Rating Product & Sorting Reviews in Amazon
####################### #################

#######################
# BUSINESS PROBLEM
#######################

# One of the most important problems in e-commerce is the correct calculation of the points given to products after sales.
# The solution to this problem means providing more customer satisfaction for the e-commerce site, making the product stand out for sellers, and a smooth shopping experience for buyers.
# Another problem is the correct ordering of the comments given to the products.
# Highlighting misleading comments will directly affect the sales of the product, causing both financial and customer loss.
# By solving these 2 basic problems, e-commerce sites and sellers will increase their sales, while customers will complete their purchasing journey without any problems.

#######################
# Dataset Story
#######################

# This dataset, which contains Amazon product data, includes product categories and various metadata.
# The product with the most comments in the electronics category has user ratings and comments.
# 12 Variable 4915 Observations 71.9 MB

# reviewerID User ID
# asin Product ID
# reviewerName Username
# helpful Helpful review rating
# reviewText Review
# overall Product rating
# summary Evaluation summary
# unixReviewTime Review time
# reviewTime Review time Raw
# day_diff Number of days since evaluation
# helpful_yes Number of times the review was found helpful
# total_vote Number of votes cast on the review

#############################
# Project Tasks
#############################
import pandas as pd
import math
import scipy.stats as st

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


####################### ####################### ###
# Task 1: Calculate the Average Rating based on current comments and compare it with the existing average rating
####################### ####################### ##
# In the shared data set, users rated a product and made comments.
# Our aim in this task is to evaluate the given points by weighting them according to date.
# It is necessary to compare the initial average score with the date-weighted score to be obtained.


# Step 1: Read the data set and calculate the average score of the product.

df=pd.read_csv("Measurement Problems/datasets/amazon_review.csv")
df["overall"].mean()
df.head()

# Step 2: Calculate the weighted score average by date.

# Declare the reviewTime variable as a date variable
# accept the max value of reviewTime as current_date
# create a new variable by expressing the difference between each score-comment date and current_date in days and
# You need to divide the variable by 4 with the quantile function (if 3 quarters are given, 4 parts are obtained) and weight it according to the values from the quarters.
# For example, if q1 = 12, when weighting, take the average of comments made less than 12 days ago and give high weight to them.

# day_diff: how many days have passed since comment

df["reviewTime"].dtype
df["reviewTime"]=pd.to_datetime(df["reviewTime"], dayfirst=True)
current_data=pd.to_datetime(str(df["reviewTime"].max()))
df["day_diff"]=(current_data - df["reviewTime"]).dt.days
df.head()
df["day_diff"].describe([0.2,0.4,0.6,0.8])
df["day_diff"].quantile(0.2)
df["day_diff"].quantile(0.4)
df["day_diff"].quantile(0.6)
df["day_diff"].quantile(0.8)
df[df["day_diff"] == df["day_diff"].quantile(0.4)][["reviewerID","reviewText","day_diff"]]

# determination of time-based average weights
def time_based_weighted_average(dataframe, w1=24, w2=22, w3=20, w4=18, w5=16):
     return dataframe.loc[dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.2),overall"].mean() * w1 / 100 + \
            dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.2)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.4)), "overall" ].mean() * w2 / 100 + \
            dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.4)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.6)), "overall" ].mean() * w3 / 100 + \
            dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.6)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.8)), "overall" ].mean() * w4 / 100 + \
            dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.8)),,"overall"].mean() * w5/100


# Step 3: Compare and interpret the average of each time period in the weighted scoring.
time_based_weighted_average(df)

####################### ####################
# Task 2: Determine 20 reviews that will be displayed on the product detail page for the product.
####################### ####################

# Step 1: Generate the helpful_no variable.
df["helpful_no"] =df["total_vote"] - df["helpful_yes"]

df= df[["reviewerName","overall","summary","helpful_yes","helpful_no","total_vote","reviewTime"]]

# Step 2: Calculate score_pos_neg_diff, score_average_rating and wilson_lower_bound scores and add them to the data.

def wilson_lower_bound(up, down, confidence=0.95):
     """
     Calculate Wilson Lower Bound Score

     - The lower limit of the confidence interval to be calculated for the Bernoulli parameter p is considered as the WLB score.
     - The score to be calculated is used for product ranking.
     - Note:
     If the scores are between 1-5, 1-3 is marked as negative and 4-5 is marked as positive and can be adapted to Bernoulli.
     This brings with it some problems. For this reason, it is necessary to make a bayesian average rating.

     parameters
     ----------
     up: int
         up count
     down: int
         down count
     confidence: float
         confidence

     returns
     -------
     wilson score: float

     """
     n = up + down
     if n == 0:
         return 0
     z = st.norm.ppf(1 - (1 - confidence) / 2)
     phat=1.0*up/n
     return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z /n)

def score_up_down_diff(up, down):
     return up - down

def score_average_rating(up, down):
     if up + down == 0:
         return 0
     return up / (up + down)

# score_pos_neg_diff
df["score_pos_neg_diff"] = df.apply(lambda x: score_up_down_diff(x["helpful_yes"],x["helpful_no"]), axis=1)

# score_average_rating
df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"],x["helpful_no"]), axis=1)

#wilson_lower_bound
df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)
df.head(20)



# Step 3: Identify 20 Comments and Interpret the results.
df.sort_values("wilson_lower_bound", ascending=False).head(20)


# • total_vote is the total number of up-downs given to a comment.
# • up means helpful.
# • There is no helpful_no variable in the data set, it must be generated from existing variables.
# • Find the number of votes that are not helpful (helpful_no) by subtracting the number of helpful votes (helpful_yes) from the total number of votes (total_vote).
# • Define the score_pos_neg_diff, score_average_rating and wilson_lower_bound functions to calculate score_pos_neg_diff, score_average_rating and wilson_lower_bound scores.
# • Create scores based on score_pos_neg_diff. Next; Save it in df with the name score_pos_neg_diff.
# • Create scores based on score_average_rating. Next; Save it in df with the name score_average_rating.
# • Create scores based on wilson_lower_bound. Next; Save it in df with the name wilson_lower_bound.
# • Identify and sort the top 20 comments according to wilson_lower_bound.
# • Interpret the results.