import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

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


# Plotting the prices through the years
plt.plot(data["Dates"],data['Prices'], label = "Actual 2021-24")
plt.plot(proj_df["Dates"],proj_df['Prices'],color='k', label = "Predicted 2024-25")
plt.ylabel('Gas Price $')
plt.xlabel("Year")
plt.title("Gas Price Forecast" , fontweight = 'bold')
plt.legend()


# A function to get the gas price in the years given or one year into the future
def get_gas_price (month, year):
    price = (round(final[(final["Month"]==month) & (final["Year"]==year)]["Prices"].iloc[0],2))
    print(f"The Gas price for the {month} month of {year} is ${price}")
    return

year= int(input("Enter the year of which you would like the price of natural gas: "))
month = int(input("Enter the month of which you would like the price of natural gas: "))

get_gas_price(month,year)
