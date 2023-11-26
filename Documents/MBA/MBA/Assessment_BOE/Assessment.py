# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 19:27:12 2023

@author: User
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# Importing and preparing data
df = pd.read_excel(r"C:\Users\User\Documents\MBA\MBA\Assessment_BOE\Data_General.xlsx")
df.describe() 


#1.Firm Size

# Rank by GWP 

 
gwp_rank = df[['Firm', 'GWP (£m)_2016YE', 'GWP (£m)_2017YE', 'GWP (£m)_2018YE',
'GWP (£m)_2019YE', 'GWP (£m)_2020YE']]
gwp_rank = gwp_rank.sort_values('GWP (£m)_2020YE', ascending=False) 

gwp_rank['Rank']= gwp_rank['GWP (£m)_2020YE'].rank(ascending= False)
gwp_rank = gwp_rank.reset_index(drop=True)

gwp_rank['Range'] = ['<100' if x <100 else '>100 < 1000' if 100<=x<1000 else '>1000 <5000' if 1000<x<5000 else '>5000' for x in gwp_rank['GWP (£m)_2020YE']]
gwp_rank['Range'].value_counts()


gwp_rank_top = gwp_rank[gwp_rank['Range'] == '>5000']

gwp_columns = [col for col in gwp_rank.columns if 'GWP (£m)_' in col]

# Plot GWP trend for each firm
plt.figure(figsize=(10, 6))
for index, row in gwp_rank_top.iterrows():
    plt.plot(range(1, len(gwp_columns) + 1), row[gwp_columns], marker='o', label=gwp_rank_top['Firm'][index])  # Assuming each row represents a firm

plt.xlabel('Years')
plt.ylabel('GWP')
plt.title('GWP Trend Over Years for Firms')
plt.xticks(range(1, len(gwp_columns) + 1), [col.split('_')[-1] for col in gwp_columns])  # Extract year from column name
plt.legend()
plt.show()

# Calculate year-over-year changes in GWP for each firm
for index, row in gwp_rank_top.iterrows():
    yoy_changes = [(row[col] - row[gwp_columns[i - 1]]) / row[gwp_columns[i - 1]] * 100 if i > 0 else 0 for i, col in enumerate(gwp_columns)]
    plt.plot(range(1, len(gwp_columns) + 1), yoy_changes, marker='o', label=gwp_rank_top['Firm'][index])

plt.xlabel('Years')
plt.ylabel('Year-over-Year GWP % Change')
plt.title('Year-over-Year GWP % Change for Firms')
plt.xticks(range(1, len(gwp_columns) + 1), [col.split('_')[-1] for col in gwp_columns])  # Extract year from column name
plt.legend()
plt.show()

#Rank by Total Assets

TA_rank = df[['Firm', 'Total assets (£m)_2020YE']]
TA_rank = TA_rank.sort_values('Total assets (£m)_2020YE', ascending=False) 

TA_rank['Rank']= TA_rank['Total assets (£m)_2020YE'].rank(ascending= False)
TA_rank = TA_rank.reset_index(drop=True)



# Calculate change percentage 
 
df['GWP_Change_2016_17'] = ((df['GWP (£m)_2017YE'] - df['GWP (£m)_2016YE'])/df['GWP (£m)_2016YE'])*100
df['GWP_Change_2017_18'] = ((df['GWP (£m)_2018YE'] - df['GWP (£m)_2017YE'])/df['GWP (£m)_2017YE'])*100
df['GWP_Change_2018_19'] = ((df['GWP (£m)_2019YE'] - df['GWP (£m)_2018YE'])/df['GWP (£m)_2018YE'])*100
df['GWP_Change_2019_20'] = ((df['GWP (£m)_2020YE'] - df['GWP (£m)_2019YE'])/df['GWP (£m)_2019YE'])*100


GWP_change = df[['Firm','GWP_Change_2016_17','GWP_Change_2017_18','GWP_Change_2018_19','GWP_Change_2019_20']]

GWP_change = GWP_change.fillna(0)

GWP_change['Cumulative_Change'] = ((1 + GWP_change['GWP_Change_2016_17'] / 100) *
                           (1 + GWP_change['GWP_Change_2017_18'] / 100) *
                           (1 + GWP_change['GWP_Change_2018_19'] / 100) *
                           (1 + GWP_change['GWP_Change_2019_20'] / 100)) - 1

# Sort the DataFrame based on cumulative year-on-year change in descending order
GWP_change_sorted = GWP_change.sort_values(by='Cumulative_Change', ascending=False)

# Select the top 10 firms based on cumulative year-on-year change
top_10_firms_GWP = GWP_change_sorted.head(10)


#Calculating Claims change
def calc_growth(x1, x2):
   if  x1 == 0:
       return np.nan
   else:
       return ((x2 - x1) / x1) * 100

df['Claims_Change_2016_17'] = df.apply(lambda x: calc_growth(x['Gross claims incurred (£m)_2016YE'], 
                                                               x['Gross claims incurred (£m)_2017YE']), axis=1)
                       
df['Claims_Change_2017_18'] = df.apply(lambda x: calc_growth(x['Gross claims incurred (£m)_2017YE'],  
                                                                x['Gross claims incurred (£m)_2018YE']), axis=1)

df['Claims_Change_2018_19'] = df.apply(lambda x: calc_growth(x['Gross claims incurred (£m)_2018YE'], 
                                                               x['Gross claims incurred (£m)_2019YE']), axis=1)
                       
df['Claims_Change_2019_20'] = df.apply(lambda x: calc_growth(x['Gross claims incurred (£m)_2019YE'],  
                                                                x['Gross claims incurred (£m)_2020YE']), axis=1)


Claims_Change = df[['Firm','Claims_Change_2016_17','Claims_Change_2017_18','Claims_Change_2018_19','Claims_Change_2019_20']]
Claims_Change = Claims_Change.fillna(0)

Claims_Change['Cumulative_Change'] = ((1 + Claims_Change['Claims_Change_2016_17'] / 100) *
                           (1 + Claims_Change['Claims_Change_2017_18'] / 100) *
                           (1 + Claims_Change['Claims_Change_2018_19'] / 100) *
                           (1 + Claims_Change['Claims_Change_2019_20'] / 100)) - 1

# Sort the DataFrame based on cumulative year-on-year change in descending order
Claims_Change_sorted = Claims_Change.sort_values(by='Cumulative_Change', ascending=False)

# Select the top 10 firms based on cumulative year-on-year change
top_10_firms = Claims_Change_sorted.head(10)

# Plotting the cumulative change for firms
plt.figure(figsize=(10, 6))  # Set figure size

# Assuming 'Firm' is a column in your DataFrame
plt.plot(top_10_firms['Firm'], top_10_firms['Cumulative_Change'], marker='o', linestyle='-', color='b')

plt.xlabel('Firms')
plt.ylabel('Cumulative Claims Change')
plt.title('Cumulative Claims Change Across Firms')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability if needed
plt.grid(True)  # Add gridlines if desired

plt.tight_layout()
plt.show()


SCR_df = df[['Firm', 'SCR (£m)_2016YE','SCR (£m)_2017YE', 'SCR (£m)_2018YE', 'SCR (£m)_2019YE',
             'SCR (£m)_2020YE', 'EoF for SCR (£m)_2016YE', 'EoF for SCR (£m)_2017YE',
             'EoF for SCR (£m)_2018YE', 'EoF for SCR (£m)_2019YE',
             'EoF for SCR (£m)_2020YE']]
             
SCR_df['SCR_Ratio (£m)_2016YE'] = (df['EoF for SCR (£m)_2016YE']/df['SCR (£m)_2016YE'])*100
SCR_df['SCR_Ratio (£m)_2017YE'] = (df['EoF for SCR (£m)_2017YE']/df['SCR (£m)_2017YE'])*100
SCR_df['SCR_Ratio (£m)_2018YE'] = (df['EoF for SCR (£m)_2018YE']/df['SCR (£m)_2018YE'])*100
SCR_df['SCR_Ratio (£m)_2019YE'] = (df['EoF for SCR (£m)_2019YE']/df['SCR (£m)_2019YE'])*100
SCR_df['SCR_Ratio (£m)_2020YE'] = (df['EoF for SCR (£m)_2020YE']/df['SCR (£m)_2020YE'])*100


# Calculate per firm
SCR_df['SCR_mean'] = SCR_df[['SCR_Ratio (£m)_2016YE', 'SCR_Ratio (£m)_2017YE',
'SCR_Ratio (£m)_2018YE', 'SCR_Ratio (£m)_2019YE',
'SCR_Ratio (£m)_2020YE']].mean(axis=1)

# Flag risky firms based on low mean  
Riskyfirms_SCR = SCR_df[SCR_df['SCR_mean'] < 100]

# Visualize the mean SCR coverage ratio for each firm
plt.figure(figsize=(10, 6))
plt.scatter(Riskyfirms_SCR.index, Riskyfirms_SCR['SCR_mean'])
plt.xlabel('Firms')
plt.ylabel('Mean SCR Coverage Ratio')
plt.title('Mean SCR Coverage Ratio for Firms')
plt.xticks(Riskyfirms_SCR.index, [f"Firm {index}" for index in Riskyfirms_SCR.Firm])  # Assuming the DataFrame index represents firms
plt.show()

Riskyfirms_SCR_filter= Riskyfirms_SCR[['Firm','SCR_mean']]

Nonriskyfirms_SCR = SCR_df[SCR_df['SCR_mean'] > 100]


NCR_df = df[['Firm','Net combined ratio_2016YE', 'Net combined ratio_2017YE',
            'Net combined ratio_2018YE', 'Net combined ratio_2019YE',
            'Net combined ratio_2020YE']]

# Calculate the mean NCR across the five years for each firm
NCR_df['Mean_NCR'] = df[['Net combined ratio_2016YE', 'Net combined ratio_2017YE',
            'Net combined ratio_2018YE', 'Net combined ratio_2019YE',
            'Net combined ratio_2020YE']].mean(axis=1)

# Rank firms based on the mean NCR (lower NCR is considered better)
NCR_df['NCR_Rank'] = NCR_df['Mean_NCR'].rank()

# Display firms ranked by their mean NCR (lower NCR gets a better rank)
ranked_firms = NCR_df.sort_values(by='NCR_Rank')
print(ranked_firms[['Firm', 'Mean_NCR', 'NCR_Rank']].head(10))  # Display top 10 ranked firms

NCR_df_filtered = NCR_df[NCR_df['Mean_NCR']>100]




#Clustering using Kmeans

from sklearn.cluster import KMeans
# Selecting columns with metrics for clustering
selected_columns = [col for col in df.columns if 'GWP (£m)_' in col or 'NWP_' in col or 'SCR_' in col or 'Gross_claims_' in col or 'Net_combined_' in col]
selected_data = df[selected_columns]

# Handling missing values or NaNs if any
selected_data.fillna(0, inplace=True)  # You can choose a different method to handle missing values

# Applying K-means clustering
kmeans = KMeans(n_clusters=3)  # Adjust the number of clusters as needed
kmeans.fit(selected_data)
selected_data['Cluster'] = kmeans.labels_

df['Cluster'] = kmeans.labels_

selected_data['Cluster'].value_counts()
# For visualization, select two columns to plot (you can change columns for other visualizations)
plt.scatter(df['GWP (£m)_2016YE'], df['Net combined ratio_2016YE'], c=df['Cluster'], cmap='viridis')
plt.xlabel('GWP Year 1')
plt.ylabel('Net Combined Ratio Year 1')
plt.title('K-means Clustering across Years')
plt.show()



#Anamoly Detection

from sklearn.ensemble import IsolationForest


# Select columns for anomaly detection (adjust based on the columns relevant to your analysis)
selected_columns = ['Firm','GWP (£m)_2016YE', 'GWP (£m)_2017YE', 'GWP (£m)_2018YE',
'GWP (£m)_2019YE', 'GWP (£m)_2020YE']

# Creating a subset of data with selected columns
selected_data_anamolies = df[selected_columns]
selected_data_anamolies_model = selected_data_anamolies[['GWP (£m)_2016YE', 'GWP (£m)_2017YE', 'GWP (£m)_2018YE','GWP (£m)_2019YE', 'GWP (£m)_2020YE']]

# Handling missing values if any
selected_data_anamolies.fillna(0, inplace=True)  # You can choose a different method to handle missing values

# Isolation Forest model
isolation_forest = IsolationForest(contamination=0.05)  # Adjust contamination (outlier fraction) as needed
isolation_forest.fit(selected_data_anamolies_model)

# Predicting outliers/anomalies (1 for normal, -1 for anomalies)
outlier_labels = isolation_forest.predict(selected_data_anamolies_model)
selected_data_anamolies['Outlier_Labels'] = outlier_labels
# Adding outlier labels to the original DataFrame
df['Outlier_Labels'] = outlier_labels

# Displaying firms marked as anomalies/outliers
anomalies = selected_data_anamolies[selected_data_anamolies['Outlier_Labels'] == -1]
print("Firms identified as anomalies:")
print(anomalies)



#Regression Model

from sklearn.linear_model import LinearRegression
import pandas as pd


# Select columns relevant for linear regression (independent and dependent variables)
independent_variables = ['GWP (£m)_2016YE', 'GWP (£m)_2017YE', 'GWP (£m)_2018YE',
'GWP (£m)_2019YE', 'GWP (£m)_2020YE']  # Adjust as needed
dependent_variable = 'Net combined ratio_2020YE'  # Adjust as needed

# Creating X (independent variables) and y (dependent variable)
X = df[independent_variables]
y = df[dependent_variable]

# Creating and fitting the linear regression model
linear_reg_model = LinearRegression()
linear_reg_model.fit(X, y)

# Coefficients and intercept of the linear regression model
coefficients = linear_reg_model.coef_
intercept = linear_reg_model.intercept_

print("Coefficients:", coefficients)
print("Intercept:", intercept)

