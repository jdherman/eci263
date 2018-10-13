import numpy as np
from scipy import stats

# 1-sample t-test

# sample size 10 from normal distribution
A = np.random.normal(500,200,10)

# null hypothesis: mu = 600
t,p = stats.ttest_1samp(A, popmean=600)

print(p)

if p < 0.05:
  print('Reject the null hypothesis')
else:
  print('Fail to reject')


# 2-sample t-test 
# does not assume same sample size or variance
# (welch's t-test)

# DE = [700, 1200, 800, 1000, 1100, 1500, 1200, 1600, 1300, 1200]
# ES = [400, 1000, 2000, 1400, 1700, 1100, 1300, 1500, 1400, 1600]

# print(np.mean(DE))
# print(np.mean(ES))

# # null hypothesis: mu_A = mu_B
# t,p = stats.ttest_ind(DE, ES, equal_var=False)

# if p < 0.05:
#   print('Reject the null hypothesis')
# else:
#   print('Fail to reject')


# Mann-Whitney U test, nonparametric, independent samples

# DE = [700, 1200, 800, 1000, 1100, 1500, 1200, 1600, 1300, 1200]*3
# ES = [400, 1000, 2000, 1400, 1700, 1100, 1300, 1500, 1400, 1600]*3

# # null hypothesis: the distributions are the same
# # alternatives: 'less', 'greater', or 'two-sided'
# U,p = stats.mannwhitneyu(DE, ES, alternative='less') 
# print(p*2)

# if p*2 < 0.05:
#   print('Reject the null hypothesis')
# else:
#   print('Fail to reject')

