import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


df = pd.read_csv('saas_spreadsheet.tsv', delimiter = '\t', header = None)
df.columns = ['user_id','payment', 'signup_month', 'payment_month']
df['payment'] = df['payment'].apply(lambda x: x.replace('$',''))
df['payment'] = df['payment'].astype('int')


df.set_index('user_id', inplace=True)
df['cohort'] = df.groupby(level=0)['signup_month'].min()
df.reset_index(inplace=True)
print df.dtypes
#groups int items
grouped = df.groupby(['cohort', 'payment_month'])
cohorts = grouped.agg({'signup_month': pd.Series.nunique, 'payment': np.sum})
cohorts.rename(columns={'payment': 'total_payment'}, inplace=True)

def cohort_period(df):
    #creates a `CohortPeriod` column, which is the nth period based on the user's first purchase.
    df['cohort_period'] = np.arange(len(df)) + 1
    return df

cohorts = cohorts.groupby(level=0).apply(cohort_period)

cohorts.reset_index(inplace=True)
cohorts.set_index(['cohort', 'cohort_period'], inplace=True)
cohort_payments = cohorts['total_payment'].unstack(0)

#spreadsheet=cohort_payments.to_csv('out.tsv', sep = '\t')

cohort_payments[[1,2,3]].plot(figsize=(10,5))
plt.title('Cohorts: Revenue')
plt.xticks(np.arange(1, 12.1, 1))
plt.xlim(1, 12)
plt.ylabel('Payment ($)')
plt.xlabel('Cohort Period (Month)')
plt.show()

sns.set(style='white')

plt.figure(figsize=(16, 8))
plt.title('Cohorts: Monthly Revenue')
sns.heatmap(cohort_payments.T, mask=cohort_payments.T.isnull(), annot=False, fmt='.0%');
plt.show()



