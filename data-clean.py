#importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#Storing data in a dataframe
df = pd.read_csv('../DATA/lending_club_loan_two.csv')
#Handling Null values
temp = pd.DataFrame(df.groupby('total_acc')['mort_acc'].mean())

def updateMort_Acc(item):
    if np.isnan( item[1] ):
        return temp.loc[item[0]]['mort_acc']
    else:
        return item[1]
        
df['mort_acc'] = df[['total_acc','mort_acc']].apply( lambda item : updateMort_Acc(item) ,axis=1)
#conversion of Categorical columns to Numeric type
def setOwnership(item):
    if item.strip() == 'NONE':
        return 'OTHER'
    elif item.strip() == 'ANY':
        return 'OTHER'
    else:
        return item

df['home_ownership'] = df['home_ownership'].apply(lambda item : setOwnership(item) )
def integerStatus(status):  # Conversion of loan status to Numeric type
    if status == 'Fully Paid':
        return 1
    else:
        return 0

def convertToInteger(item):   # Conversion of loan term to Numeric type
    if item.strip() == '36 months':
        return 36
    else:
        return 60

df['earliest_cr_year'] = df['earliest_cr_line'].apply(lambda date : int(date.split('-')[-1]) )
df['zip_code'] = df['address'].apply(lambda address : address.split()[-1] )
df['term'] = df['term'].apply(lambda item : convertToInteger(item) )
df['loan_repaid'] = df['loan_status'].apply( integerStatus )

#Conversion of Categorical columns with more than 2 distinct values to dummy variables
sub_grade_dummies           =   pd.get_dummies(df['sub_grade'],dtype=int,drop_first=True)
verification_status_dummies =   pd.get_dummies(df['verification_status'],dtype=int,drop_first=True)
application_type_dummies    =   pd.get_dummies(df['application_type'],dtype=int,drop_first=True)
initial_list_status_dummies =   pd.get_dummies(df['initial_list_status'],dtype=int,drop_first=True)
purpose_dummies             =   pd.get_dummies(df['purpose'],dtype=int,drop_first=True)
home_ownership_dummies      =   pd.get_dummies(df['home_ownership'],drop_first=True,dtype=int)
zip_code_dummies            =   pd.get_dummies(df['zip_code'],drop_first=True,dtype=int)


#Dropping attributes that are either converted to dummy variables or does not have any effect on the outcome of the ML model. 
df.drop(['emp_title','emp_length','title','grade','sub_grade','verification_status','application_type','initial_list_status',
         'purpose','home_ownership','address','zip_code','issue_d','earliest_cr_line''loan_status'],axis=1,inplace=True)
