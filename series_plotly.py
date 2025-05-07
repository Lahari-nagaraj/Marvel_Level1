import pandas as pd
import numpy as np
import plotly.express as px
import webbrowser

# Step 1: Generate simulated stock data
np.random.seed(0)

# Create date range
dates = pd.date_range('2022-01-01', periods=500, freq='B')  # 'B' for business days (weekdays only)

# Simulate stock prices with a random walk (random fluctuations)
price_changes = np.random.normal(0, 1, len(dates))  # Mean=0, Std=1
prices = 100 + np.cumsum(price_changes)  # Starting at 100, cumulative sum of price changes

# Create a DataFrame
stock_data = pd.DataFrame({'Date': dates, 'Price': prices})

# Step 2: Plot the time series using Plotly
fig = px.line(stock_data,
              x='Date',
              y='Price',
              title='Simulated Stock Prices (2022–2024)',
              labels={'Price': 'Stock Price (USD)'},
              template='plotly_dark')

# Step 3: Save and open the plot in the browser
fig.write_html("simulated_stock_prices.html")
webbrowser.open("simulated_stock_prices.html")
