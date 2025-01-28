import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime

#Import data from the csv file
data = pd.read_csv('Nat_Gas.csv')

#Adjust the data by adding year and month columns
data["Dates"] = pd.to_datetime(data["Dates"])
data["Month"]= data["Dates"].dt.month
data["Year"] = data["Dates"].dt.year


# Extrapolating the Prices for each month for one year into the future using linear regression
def ext_year_price(year):
    new_yr_prices =[11.80]
    # Extrapolating for the remaining months of 2024
    for i in range(9,12):
        X=np.array(data[data["Month"]==1+i]["Year"]).reshape((-1,1))
        Y=np.array(data[data["Month"]==1+i]["Prices"])
        tempmodel = LinearRegression().fit(X,Y)
        new_yr_prices.append(tempmodel.predict([[2024]])[0])
    # Extrapolating for the months of 2025
    for i in range(12):
        X=np.array(data[data["Month"]==1+i]["Year"]).reshape((-1,1))
        Y=np.array(data[data["Month"]==1+i]["Prices"])
        tempmodel = LinearRegression().fit(X,Y)
        new_yr_prices.append(tempmodel.predict([[year]])[0])
    return new_yr_prices

# Creating an array filled with the dates of the corresponding predicted prices 
def get_last_day(year):
    dates = pd.date_range(start=f"{2024}-09-01", end=f"{year}-12-31", freq='M')
    return dates

# Creating a new data frame of the new projected prices
proj_df = pd.DataFrame({"Dates": get_last_day(2025), "Prices":ext_year_price(2025)})
proj_df["Dates"] = pd.to_datetime(proj_df["Dates"])
proj_df["Month"]= proj_df["Dates"].dt.month
proj_df["Year"] = proj_df["Dates"].dt.year

# Combining the data frames of the given data and the predicted prices
final = pd.concat([data, proj_df], ignore_index=True)




# A function to extrapolate or retreive the gas price for any date
def predict_gas_price(date):
    input_date = pd.to_datetime(date,dayfirst=True)
    if input_date in final["Dates"].values:
        return round(final.loc[final["Dates"]== input_date,"Prices"].values[0],2)
    else:
        input_month = input_date.month
        input_yr = input_date.year
        X=np.array(final[data["Month"]==input_month]["Year"]).reshape((-1,1))
        Y=np.array(data[data["Month"]==input_month]["Prices"])
        tempmodel = LinearRegression().fit(X,Y)
        return round(tempmodel.predict([[input_yr]])[0],2)


# Function which calculates the value of a pre-specified contract
def calc_contract_value(injection_dates,withdrawal_dates, injection_rate, injection_withdrawal_costs,max_volume, storage_costs):
    tot_value=0
    volume = 0
    i=j=0
    injection_dates_adj = [datetime.strptime(date, "%d/%m/%Y") for date in injection_dates]
    withdrawal_dates_adj = [datetime.strptime(date, "%d/%m/%Y") for date in withdrawal_dates]
    while (i < len(injection_dates) or j< len(withdrawal_dates)):
        if i < len(injection_dates) and (j >= len(withdrawal_dates) or injection_dates[i] > withdrawal_dates[j]):
            injection_price= predict_gas_price(injection_dates[i])
            print("Injection price =$",injection_price)
            injectable_volume = min(injection_rate, max_volume - volume)
            print("Injected:",injectable_volume)
            volume += injectable_volume
            tot_value-=(injectable_volume *injection_price) +injection_withdrawal_costs
            i+=1
        else:
            withdrawal_price= predict_gas_price(withdrawal_dates[j])
            print("Withdrawal price =$",withdrawal_price)
            withdrawable_volume = min(injection_rate, volume)
            print("Withdrawed:",withdrawable_volume)
            volume -= withdrawable_volume
            tot_value+=withdrawable_volume *withdrawal_price -injection_withdrawal_costs
            j+=1
    tot_months = (withdrawal_dates_adj[-1].year - injection_dates_adj[0].year) * 12 + (withdrawal_dates_adj[-1].month - injection_dates_adj[0].month)
    tot_value -= (tot_months*storage_costs)
    print("Total Value of Contract: $",tot_value)
    return

# Takes required inputs and calculates the contract value
print("NOTE: Keep units consistent")
injection_dates =[]
injection_times = int(input("Enter the number of times/dates you would like to inject gas: "))
for i in range(injection_times):
    date_entry = input(f'Enter {i+1} injection date in DD/MM/YYYY format')
    injection_dates.append(date_entry)

withdrawal_dates =[]
withdrawal_times = int(input("Enter the number of times/dates you would like to withdraw gas: "))
for i in range(withdrawal_times):
    date_entry = input(f'Enter withdrawal date {i + 1} in DD/MM/YYYY format: ')
    withdrawal_dates.append(date_entry)
injection_rate = int(input("Enter the injection/withdrawal rate: "))
injection_withdrawal_costs = int(input("Enter the injection/withdrawal cost: "))
max_volume=int(input("Enter the max volume of storage available: "))
storage_costs = int(input("Enter the storage cost/month: "))
calc_contract_value(injection_dates,withdrawal_dates, injection_rate,injection_withdrawal_costs,max_volume,storage_costs)


