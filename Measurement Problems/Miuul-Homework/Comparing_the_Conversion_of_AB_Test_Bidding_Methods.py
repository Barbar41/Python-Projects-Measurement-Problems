######
# Comparing AB Test and Conversion of Bidding Methods
######

#######################
# BUSINESS PROBLEM
#######################
# Facebook recently introduced a new type of bidding, "average bidding", as an alternative to the existing type of bidding called "maximumbidding".
# One of our customers, bombabomba.com, decided to test this new feature and found that average bidding was replaced by maximumbidding /
# Wants to run an A/B test to see if it brings more conversions.
# A/B testing has been going on for 1 month and bombabomba.com now expects you to analyze the results of this A/B test. The ultimate success criterion for Bombomba.com is Purchase.
# Therefore, the focus should be on the Purchase metric for statistical testing.

#######################
# Dataset Story
#######################
# This data set, which includes a company's website information, includes information such as the number of advertisements that users see and click on, as well as earnings information obtained from it.
# There are two separate data sets: Control and Test group. These data sets are located on separate sheets of the ab_testing.xlsx excel.
# Maximum Bidding was applied to the control group and Average Bidding was applied to the test group.

#4 Variables 40 Observations 26 KB
# Impression: Number of ad views
# Click: Number of clicks on the displayed ad
# Purchase: Number of products purchased after ads clicked
# Earning: Earnings earned after purchasing products

#############################
# Project Tasks
#############################

####################
# Task 1: Preparing and Analyzing Data
####################
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


# Step 1: Read the data set consisting of control and test group data named ab_testing_data.xlsx. Assign control and test group data to separate variables.

dataframe_control=pd.read_excel("Measurement Problems/datasets/ab_testing.xlsx", sheet_name="Control Group")
dataframe_test=pd.read_excel("Measurement Problems/datasets/ab_testing.xlsx", sheet_name="Test Group")

df_control= dataframe_control.copy()
df_test=dataframe_test.copy()

# Step 2: Analyze control and test group data.

def check_df(dataframe, head=5):
     print("################################## Shape #################")
     print(dataframe.shape)
     print("################################## Types #################")
     print(dataframe.dtypes)
     print("################################## Head ##################")
     print(dataframe.head())
     print("################################## Tail ################")
     print(dataframe.tail())
     print("##################################NA ##################")
     print(dataframe.isnull().sum())
     print("################################## Quantiles #################")
     print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99,1]).T)

check_df(df_control)

check_df(df_test)

# Step 3: After the analysis, combine the control and test group data using the concat method.

df_control["group"]= "control"
df_test["group"]="test"

df=pd.concat([df_control,df_test], axis=0, ignore_index=False)
df.head()


####################
# Task 2: Defining the Hypothesis of A/B Testing
####################

# Step 1: Define the hypothesis.
# H0 : M1 = M2 (There is no difference between the purchasing averages of the control group and test group)
# H1 : M1!= M2 (There is a difference between the purchasing averages of the control group and the test group)

# Step 2: Analyze the purchase (earnings) averages for the control and test groups.

df.groupby("group").agg({"Purchase":"mean"})

####################
# Task 3: Performing Hypothesis Testing
####################

# Step 1: Perform assumption checks before hypothesis testing.
# ------------------------------------------------- -------------------------------------------------- --------
# These are Normality Assumption and Homogeneity of Variance. Test separately whether the control and test groups comply with the normality assumption using the Purchase variable.
# Normality Assumption:
# H0: Normal distribution assumption is met.
# H1: Normal distribution assumption is not met.
# p < 0.05 H0 REJECT, p > 0.05 H0 CANNOT BE REJECTED
# According to the test result, is the assumption of normality met for the control and test groups? Interpret the p-value values obtained.

# Variance Homogeneity:
# H0: Variances are homogeneous.
# H1: Variances are not homogeneous.
# p < 0.05 H0 REJECT, p > 0.05 H0 CANNOT BE REJECTED
# Test whether variance homogeneity is achieved for the control and test groups using the Purchase variable.
# Is the assumption of normality met based on the test result? Interpret the p-value values obtained.

test_stat, pvalue = shapiro(df.loc[df["group"] == "control", "Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value=0.5891 (HO CANNOT BE REJECTED. THE VALUES OF THE CONTROL GROUP SATISFY THE NORMAL DISTRIBUTION ASSUMPTION)

test_stat, pvalue = shapiro(df.loc[df["group"] == "test", "Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value=0.1541 (HO CANNOT BE REJECTED. THE VALUES OF THE TEST GROUP SATISFY THE NORMAL DISTRIBUTION ASSUMPTION)

# Since normal distribution is ensured, we proceed to the variance homogeneity test.
# If it was not provided, we would do nonparametric testing.
# Variance Homogeneity:
# Variance is the sum of the squares of the deviations of the data from the arithmetic mean. In other words, it is the square root of the standard deviation.
# H0: Variances are homogeneous.
# H1: Variances are not homogeneous.
# p < 0.05 H0 REJECT, p > 0.05 H0 CANNOT BE REJECTED

test_stat, pvalue = levene(df.loc[df["group"] == "control", "Purchase"],
                            df.loc[df["group"] == "test", "Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value=0.1083 (HO CANNOT BE REJECTED. THE VALUES OF THE CONTROL AND TEST GROUP PROVIDE THE ASSUMPTION OF VARIANCE HOMOGENEITY)
# Since Variance Homogeneity is ensured, we will proceed to the independent two sample T test.
# If variance homogeneity was not ensured, we would apply it to our variable where only one argument's equal_var is False for variance homogeneity.


# Step 2: Select the appropriate test according to the Normality Assumption and Variance Homogeneity results.

# Since the variances are provided, an independent two-sample T test (parametric test) is performed.
# H0: M1 = M2 (There is no statistically significant difference between the purchasing averages of the control group and the test group)
# H1:M1! = M2 (not equal) (There is a Statistically Significant Difference Between the purchasing averages of the control group and the test group)
# M1 and M2=Representations of the population average.
# p < 0.05 H0 REJECT, p > 0.05 H0 CANNOT BE REJECTED

test_stat, pvalue = ttest_ind(df.loc[df["group"] == "control", "Purchase"],
                               df.loc[df["group"] == "test", "Purchase"],
                               equal_var=True)

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Step 3: Considering the p_value value obtained as a result of the test, comment on whether there is a statistically significant difference between the purchasing averages of the control and test groups.

# p-value=0.3493 (HO CANNOT BE REJECTED. THERE IS NO STATISTICALLY SIGNIFICANT DIFFERENCE BETWEEN THE PURCHASE AVERAGES OF THE CONTROL AND TEST GROUP)



####################
# Task 4: Analysis of Results
####################

# Step 1: State which test you used and the reasons.
# First, a normality test was applied to both groups. Since it was observed that both groups followed normal distribution,
# Moving to the second assumption, the homogeneity of variance was examined.
# Since the variances were homogeneous, "Independent Two Sample T Test" was applied.
# As a result of the application, it was observed that the p-value was greater than 0.05 and the H0 hypothesis could not be rejected.


# Step 2: Give advice to the customer based on the test results you obtain.
# Since there is no significant difference in terms of purchase, the customer can choose one of the two methods. However, differences in other statistics will also be important here.
# Differences in clicks, interactions, earnings and conversion rates can be evaluated and it can be determined which method is more profitable.
# Especially since Facebook is paid per click, it is determined which method has a lower click rate and the CTR rate can be looked at and two groups continue to be observed.