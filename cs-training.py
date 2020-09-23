"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Evaluating credit risk for individuals as a financial institution

The dataset contains a sample of 150.000 clients of a financial institution. Description of columns:

- SeriousDlqin2yrs => Dummy variable to represent clients that experienced 90 days or more of delay in payment.
This variable has 6.7% positive cases and 93.3% negative cases, meaning that it's clearly unbalanced.  
- RevolvingUtilizationOfUnsecuredLines => Total balance on credit cards plus personal lines of credit in relation
to the sum of credit limits.
- Age => Age of borrower in years.
- NumberOfTime30-59DaysPastDueNotWorse => Number of times the borrower has been 30-59 days past due in the last
2 years.
- DebtRatio => Monthly debt payments in relation to monthly gross income.
- MonthlyIncome => Monthly income.
- NumberOfOpenCreditLinesAndLoans => Number of open loans and lines of credit.
- NumberOfTimes90DaysLate => Number of times borrower has been 90 days or more past due
- NumberRealEstateLoansOrLines => Number of mortgage and real estate loans.
- NumberOfTime60-89DaysPastDueNotWorse => Number of times the borrower has been 60-80 days past due in the last
2 years.
- Number of Dependents => Number of dependents in family.

In order to extract random variables to train and test a sample, we should re-balance it to have equal chances to
obtain positive and negative cases (stratified samplings). This re-sampling is independent from the folding of the
sample to cross-validate training sets.

The problem we're facing is a classification one, as we have to predict the class of client's of the bank, which
consists of DEFAULT (1) or NO DEFAULT (0). Although intuition might say that I should not borrow to clients that
were classified as 1, it's important to consider that in order to 'teach' my model how parameters are changing
I should borrow to clients that presumably won't pay me.

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# IMPORTING LIBRARIES
import pandas as pd
import numpy as np
import Orange
from Orange.data import Domain, Table
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# CREATING DATAFRAME WITH SAMPLE DATA
df = pd.read_csv('cs-training.csv')
df.columns

# PLOTTING
# Delinquency values
x_bar_labels = ['Not Default', 'Default']
x_values = list(range(len(x_bar_labels)))
y_values1 = sum(df[df.columns[1]])
y_values0 = int(df[df['SeriousDlqin2yrs'] == 0]['SeriousDlqin2yrs'].count())
y_values = [y_values0, y_values1]

fig = plt.figure(figsize=(8,6))
ax = fig.subplots()
plt.bar(x_values, y_values,facecolor='peru', edgecolor='blue')
plt.ylabel('Delincuencies')
ax.set_xticks([0,1])
ax.set_xticklabels(x_bar_labels)
plt.title("Example image for Imbalanced Datasets")
plt.savefig('bar_plot_defaults')
plt.show()

# Credit card spending 
x_values1 = df.loc[(df['RevolvingUtilizationOfUnsecuredLines'] >= 0) & (df['RevolvingUtilizationOfUnsecuredLines'] < 3)]
x_values1['RevolvingUtilizationOfUnsecuredLines'].describe()
fig2 = plt.figure(figsize=(8,6))
ax2 = fig2.subplots()
plt.hist(x_values1['RevolvingUtilizationOfUnsecuredLines'],facecolor='peru', edgecolor='blue',bins=100)
plt.title('Credit expenses as a ratio of spending capacity distribution')
plt.xlim([0,1.2])
plt.ylabel('Count')
plt.savefig('hist_credit_exp')
plt.show()

# Debt ratio
debt_values = df.loc[(df['DebtRatio'] >= 0) & (df['DebtRatio'] < 3)]
debt_values['DebtRatio'].describe()
fig_debt = plt.figure(figsize=(8,6))
ax3 = fig_debt.subplots()
plt.hist(debt_values['DebtRatio'],bins=100,facecolor='peru', edgecolor='blue')
plt.title('Debt Ratio')
plt.xlim([0,1.2])
plt.ylabel('Count')
plt.savefig('debt_ratio')
plt.show()
