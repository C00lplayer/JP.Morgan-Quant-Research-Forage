import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Opening CSV file
data= pd.read_csv("Task 3 and 4_Loan_Data.csv")

# Function to find squared error for a specific bucket
def find_squared_error(fico_bucket):
    ni= len(fico_bucket)
    ki = fico_bucket["default"].sum()
    mse = ((fico_bucket["default"]-(ki/ni))**2).sum()
    return mse


# Function to find MSE for specific bucketing
def find_mse(infodf):
    total_sum_mse = 0
    for bucket in range(infodf["buckets"].max()+1):
        bucket_data = infodf[infodf["buckets"]==bucket]
        total_sum_mse+= find_squared_error(bucket_data)
    return total_sum_mse/len(infodf)


# Function to find the lowest MSE depening on the number of buckets
def optimise_buckets(data_df):
    mean_mse_vals= []
    for i in range(2,15):
        data_df["buckets"]= pd.qcut(data_df["fico_score"], q = i, labels = False).astype(int)
        mean_mse = find_mse(data_df)
        mean_mse_vals.append((mean_mse,i))
    min_mean_mse = min(mean_mse_vals, key= lambda x:x[0])

    # Visulaising Number of Buckets vs. MSE 
    
    """
    bucket_counts, mse_values = zip(*mean_mse_vals)
    plt.plot(mse_values, bucket_counts, marker='o')
    plt.xlabel("Number of Buckets")
    plt.ylabel("Mean Squared Error")
    plt.title("Number of Buckets vs. MSE")
    plt.show()
    """
    print(f"Optimal Buckets = {min_mean_mse[1]} and MSE = {min_mean_mse[0]}")
    return
    



optimise_buckets(data)
    