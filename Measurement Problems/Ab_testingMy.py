####################### ####
# Basic Statistics Concepts
####################### ####

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# !pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
     pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

#############################
# Sampling
#############################

# We create our number population
population = np.random.randint(0, 80, 10000)
population.mean()

# random seed setting to draw samples
np.random.seed(115)

# To select sample
sample = np.random.choice(a=population, size=100)
example.mean()

# We sample and select
np.random.seed(10)
sample1 = np.random.choice(a=population, size=100)
sample2 = np.random.choice(a=population, size=100)
sample3 = np.random.choice(a=population, size=100)
sample4 = np.random.choice(a=population, size=100)
sample5 = np.random.choice(a=population, size=100)
sample6 = np.random.choice(a=population, size=100)
sample7 = np.random.choice(a=population, size=100)
sample8 = np.random.choice(a=population, size=100)
sample9 = np.random.choice(a=population, size=100)
sample10 = np.random.choice(a=population, size=100)

# Take the average of the samples
(example1.mean() + example2.mean() + example3.mean() + example4.mean() + example5.mean()
  + example6.mean() + example7.mean() + example8.mean() + example9.mean() + example10.mean()) / 10


#############################
# Descriptive Statistics
#############################

# we are pulling tips dataset from seabor
df = sns.load_dataset("tips")
df.describe().T

df.head()

# Calculating the confidence interval for the relevant variable
sms.DescrStatsW(df["total_bill"]).tconfint_mean()

# A review for future tips
sms.DescrStatsW(df["type"]).tconfint_mean()

# Confidence Interval Calculation for Numerical Variables in Titanic Data Set
df = sns.load_dataset("titanic")
df.describe().T

# Calculating the confidence interval after removing the missing values for the relevant variable
sms.DescrStatsW(df["age"].dropna()).tconfint_mean()

# Calculating the confidence interval after removing the missing values for the relevant variable
sms.DescrStatsW(df["mouse"].dropna()).tconfint_mean()

####################### ####
# Correlation
####################### ####

# Tip dataset:
# total_bill: total price of the meal (including tip and tax)
# type: tip
# sex: gender of the person paying the fee (0=male, 1=female)
# smoker: is there anyone in the group who smokes? (0=No, 1=Yes)
# day: day (3=Thur, 4=Fri, 5=Sat, 6=Sun)
# time: when? (0=Day, 1=Night)
# size: how many people are in the group?

df = sns.load_dataset('tips')
df.head()


df["total_bill"] = df["total_bill"] - df["type"]

# Let's examine the relationship between the two with the scatter plot (there is a positive, moderate relationship).
df.plot.scatter("type", "total_bill")
plt.show()

# Mathematical equivalent (slightly above moderate intensity) (we use it to observe the correlation between two variables)
df["type"].corr(df["total_bill"])

#############################
# Application 1: Is there a statistical difference between the account averages of smokers and non-smokers?
#############################

df = sns.load_dataset("tips")
df.head()

# Let's look at the averages of the two groups (according to smoking or non-smoking status)
df.groupby("smoker").agg({"total_bill": "mean"})
# It seems like there is a difference, but let's look at it statistically.


#############################
# 1. Establish Hypothesis
#############################

# H0: M1 = M2(
# H1: M1 != M2

#############################
#2. Assumption Checking
#############################

# Normality Assumption: It is a hypothesis test of whether the distribution of a variable is similar to the standard normal distribution.
# Variance Homogeneity:

#############################
# Normality Assumption
#############################

# H0: Normal distribution assumption is met.
# H1:..not provided.

# For smokers to check normal distribution
test_stat, pvalue = shapiro(df.loc[df["smoker"] == "Yes", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# HO RED from 0.05 if p-value <.
# If p-value is not < 0.05 H0 CANNOT BE REJECTED.

# For non-smokers to check normal distribution
test_stat, pvalue = shapiro(df.loc[df["smoker"] == "No", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Both did not comply with the normal distribution. H0 REJECTED

# Since the Normality Assumption is not satisfied, we must use a parametric test.

#############################
# Homogeneity of Variance Assumption
#############################

# H0: Variances are Homogeneous
# H1: Variances Are Not Homogeneous

# It checks for homogeneity of variance according to two different groups with the Levene test.

test_stat, pvalue = levene(df.loc[df["smoker"] == "Yes", "total_bill"],
                            df.loc[df["smoker"] == "No", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Variance Homogeneity was not achieved and this was also rejected.

# HO RED if p-value < 0.05.
# If p-value is not < 0.05 H0 CANNOT BE REJECTED.


#############################
#3 and 4. Application of Hypothesis
#############################

# 1. If the assumptions are met, independent two-sample t-test (parametric test)
# 2. Mannwhitneyu test (non-parametric test) if the assumptions are not met

#############################
# 1.1 Independent two-sample t-test (parametric test) if assumptions are met
#############################

# The t test method that can be used if the assumption of normality is met.
# It is entered only if variance homogeneity is not ensured (equal_var=False).

test_stat, pvalue = ttest_ind(df.loc[df["smoker"] == "Yes", "total_bill"],
                               df.loc[df["smoker"] == "No", "total_bill"],
                               equal_var=True)

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# HO RED if p-value < 0.05.
# If p-value is not < 0.05 H0 CANNOT BE REJECTED.

#############################
# 1.2 Mannwhitneyu test if assumptions are not met (non-parametric test)
#############################
# HO could not be rejected.

# nunprametric mean benchmark median benchmark
test_stat, pvalue = mannwhitneyu(df.loc[df["smoker"] == "Yes", "total_bill"],
                                  df.loc[df["smoker"] == "No", "total_bill"])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))



#############################
# Application 2: Statistical Significance Between the Average Ages of Titanic Female and Male Passengers. Difference. are there?
#############################

df = sns.load_dataset("titanic")
df.head()

df.groupby("sex").agg({"age": "mean"})
# There seems to be a difference. Could this difference have occurred by chance?


# 1. Establish hypotheses:
# H0: M1 = M2 (There is no statistically significant difference between the average ages of female and male passengers)
#H1:M1! = M2 (not equal) (There is a statistically significant difference between the average ages of female and male passengers)
# M1 and M2=Representations of the population average.


# 2. Examine Assumptions

# Normality assumption
# H0: Normal distribution assumption is met.
# H1:..not provided

test_stat, pvalue = shapiro(df.loc[df["sex"] == "female", "age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue)) #HO:REJECTED

test_stat, pvalue = shapiro(df.loc[df["sex"] == "male", "age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue)) #HO:REJECTED

# No assumptions are made for the two groups.

# Homogeneity of variance
# H0: Variances are Homogeneous
# H1: Variances Are Not Homogeneous

test_stat, pvalue = levene(df.loc[df["sex"] == "female", "age"].dropna(),
                            df.loc[df["sex"] == "male", "age"].dropna())

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# H0: We could not reject it because it was not small. This means that the variances are homogeneous.

# Nonparametric as assumptions are not met

test_stat, pvalue = mannwhitneyu(df.loc[df["sex"] == "female", "age"].dropna(),
                                  df.loc[df["sex"] == "male", "age"].dropna())

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# H0: Reject, comment. What we observed between the age averages of male and female passengers
# the difference is also statistically significant. There is a difference and we saw it!

#90 280


#############################
# Application 3: Average Ages of People with and without Diabetes. Between Ist. Be. Anl. Is there any difference?
#############################

df = pd.read_csv("Measurement Problems/datasets/diabetes.csv")
df.head()

df.groupby("Outcome").agg({"Age": "mean"})

# 1. Set up hypotheses
#H0:M1=M2
# Average Ages of People with and without Diabetes. Between Ist. Be. Anl. There Is No Difference
# H1: M1 != M2
# .... has.

#2. Examine Assumptions

# Normality Assumption (H0: Normal distribution assumption is met.)
test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 1, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 0, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))



# Nonparametric because normality assumption is not met.
# Output as median comparison or ranking comparison
# H0: rejected.

# Hypothesis (H0: M1 = M2)
test_stat, pvalue = mannwhitneyu(df.loc[df["Outcome"] == 1, "Age"].dropna(),
                                  df.loc[df["Outcome"] == 0, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


####################### #
# Business Problem: Are the scores of those who watched the majority of the course different from those who did not?
####################### #

# H0: M1 = M2 (... there is no significant difference between the two group averages.)
# H1: M1 != M2 (...exists)

df = pd.read_csv("Measurement Problems/datasets/course_reviews.csv")
df.head()

# Trainees whose progress is more than 75%
df[(df["Progress"] > 75)]["Rating"].mean()

# Trainees whose progress is more than 25%
df[(df["Progress"] < 25)]["Rating"].mean()


test_stat, pvalue = shapiro(df[(df["Progress"] > 75)]["Rating"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


test_stat, pvalue = shapiro(df[(df["Progress"] < 25)]["Rating"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


test_stat, pvalue = mannwhitneyu(df[(df["Progress"] > 75)]["Rating"],
                                  df[(df["Progress"] < 25)]["Rating"])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

####################### ####
# AB Testing (Two Sample Proportion Test)
####################### ####

#H0:p1=p2
# There is no statistically significant difference between the conversion rate of the new design and the conversion rate of the old design.
# H1: p1 != p2
# ... has

# We put both groups in separate arrays
number_successes = np.array([300, 250])
observation_number = np.array([1000, 1100])

# We perform the process using the comparison method
proportions_ztest(count=success_sayisi, nobs=observation_counts)
# We reject H0
# H1 THERE IS A STATISTICAL DIFFERENCE BETWEEN THE TWO RATIOS.

number of successes / number of observations

#############################
# Application: Is There a Statistically Significant Difference Between the Survival Rates of Men and Women?
#############################

# H0: p1 = p2 (p1-p2=0)
# No Statistically Significant Difference Between Survival Rates of Women and Men

# H1: p1 != p2
# .. has

df = sns.load_dataset("titanic")
df.head()

df.loc[df["sex"] == "female", "survived"].mean()

df.loc[df["sex"] == "male", "survived"].mean()

# For these proportions, the first argument for the proportions test requires the number of successes and the second argument requires the number of observations.

female_succ_count = df.loc[df["sex"] == "female", "survived"].sum()
male_succ_count = df.loc[df["sex"] == "male", "survived"].sum()

test_stat, pvalue = proportions_ztest(count=[female_succ_count, male_succ_count],
                                       nobs=[df.loc[df["sex"] == "female", "survived"].shape[0],
                                             df.loc[df["sex"] == "male", "survived"].shape[0]])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# We evaluated the difference between the survival rates of men and women.
# H0 Rejected. There is no significant difference
# H1 There is a significant difference.

####################### ####
# ANOVA (Analysis of Variance)
####################### ####

# Used to compare more than two group averages.

df = sns.load_dataset("tips")
df.head()

df.groupby("day")["total_bill"].mean()

# 1. Set up hypotheses

# HO: m1 = m2 = m3 = m4
# There is no difference between group averages.

# H1: .. there is a difference

#2. Assumption check

# Normality assumption
# Assumption of homogeneity of variance

# If the assumption is met, one way anova
# If the assumption is not met, kruskal

# H0: Normal distribution assumption is met.

# Let's do such an operation, let's filter the day variable in the dataset and then do the Shapiro test.
# We turned it into an iterative object that can navigate over the classes of a categorical variable (to a list).

for group in list(df["day"].unique()):
     pvalue = shapiro(df.loc[df["day"] == group, "total_bill"])[1]
     print(group, 'p-value: %.4f' % pvalue)

# --H0 is rejected because the p value is less than 0.05 for all of them.
# --No normality assumption is provided for any of them:


# H0: The assumption of variance homogeneity is satisfied.
# Levene method for 4 groups

test_stat, pvalue = levene(df.loc[df["day"] == "Sun", "total_bill"],
                            df.loc[df["day"] == "Sell", "total_bill"],
                            df.loc[df["day"] == "Thur", "total_bill"],
                            df.loc[df["day"] == "Fri", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# H0 irrefutable variance homogeneity is ensured.


#3. Hypothesis testing and p-value interpretation

# None of them provide it.
df.groupby("day").agg({"total_bill": ["mean", "median"]})

# There is a difference in medians, we do not know whether it is significant or not.
# Testing whether there is a difference between these groups and testing the averages of the groups are different things.

# HO: There is no statistically significant difference between group averages

# If the assumption is met, we will use one-way parametric testing
# parametric anova test:
f_oneway(df.loc[df["day"] == "Thur", "total_bill"],
         df.loc[df["day"] == "Fri", "total_bill"],
         df.loc[df["day"] == "Sat", "total_bill"],
         df.loc[df["day"] == "Sun", "total_bill"])
# H0 with a p value less than 0.05 is rejected.
# Naturally, there is a significant difference between the groups.
# These assumptions were not met, let's use nonparamatric instead of parametric
# Nonparametric anova test:
 kruskal(df.loc[df["day"] == "Thur", "total_bill"],
                 df.loc[df["day"] == "Fri", "total_bill"],
                 df.loc[df["day"] == "Sell", "total_bill"],
                 df.loc[df["day"] == "Sun", "total_bill"])

# H0 with p value less than 0.05 is rejected.
# We see that there is a significant difference between these groups.
# # Yes, there is a difference, but the difference comes from who, what and which group.

from statsmodels.stats.multicomp import MultiComparison
comparison = MultiComparison(df['total_bill'], df['day'])
tukey = comparison.tukeyhsd(0.05)
print(tukey.summary())

# Statistically, no significant difference was found by pairwise comparison. But when we looked at the group as a whole, there was a difference.
# The alpha value can be changed and adjusted for a comparable value.
# It can be assumed that there is no difference.











