# -*- coding: utf-8 -*-
"""
Created on Wed May 17 03:24:09 2023

@author: Damion Chau
"""

import pandas as pd

# Read the loan data from the CSV file
loan_data = pd.read_csv('hmeq.csv')

# Define a function to analyze the loan data and identify risk factors
def analyze_loan_data(data):
    # Calculate the default rate for each loan type
    default_rates = data.groupby('LOAN')['BAD'].mean()

    # Determine the risk level for each loan type based on the default rate
    risk_levels = pd.cut(default_rates, bins=[0, 0.1, 0.2, 1], labels=['Low Risk', 'Medium Risk', 'High Risk'])

    # Add a new column to the loan data with the risk level for each loan type
    data['Risk_Level'] = data['LOAN'].map(risk_levels)

    return data

# Analyze the loan data and get the results
loan_data = analyze_loan_data(loan_data)

# Apply the risk level recommendations based on the analysis results
loan_data['Risk_Recommendation'] = ''
loan_data.loc[loan_data['Risk_Level'] == 'Low Risk', 'Risk_Recommendation'] = 'Approval'
loan_data.loc[loan_data['Risk_Level'] == 'Medium Risk', 'Risk_Recommendation'] = 'Conditional approval'
loan_data.loc[loan_data['Risk_Level'] == 'High Risk', 'Risk_Recommendation'] = 'Reject'

# Write the results to a new CSV file with the new columns added
loan_data.to_csv('loan_data_analysis.csv', index=False)

# Print a summary of the analysis results and risk level recommendations
print('Loan Data Analysis Results:')
print('---------------------------')
print('Total number of loans:', len(loan_data))
print('Number of loans in each risk category:')
print(loan_data['Risk_Level'].value_counts())
print('Risk level recommendations for each loan:')
print(loan_data[['LOAN', 'Risk_Level', 'Risk_Recommendation']])