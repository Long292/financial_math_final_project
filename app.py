import streamlit as st
import pandas as pd
# import math
import requests
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf

# Define variables
# total_years = 51
# interest_rate = 0.08
# withdrawal_year_23 = 850  # in MVND
# withdrawal_increase = 0.04

# # Title andの説明 (description in Japanese)
# st.title("A person decides to deposit a constant amount P at the begin of each of the next 21 years, with a constant interest rate 8% per year compounded yearly. For year 23 th, the person desires to withdraw 850 MVND for spending. For the years after (from year 24 th) until year 51 th, the yearly amount withdrew increases by 4% annually to avoid the effect of inflation. Note that no money left over after year 51 th. To obtain this goal, what is the value of P in MVND?")
# # st.write("毎年の年初に一定額を預金し、23年目以降は毎年引き出し額を増やしていく場合の将来の貯蓄額をシミュレーションします。")

# # Input for P value
# p_value = st.number_input("Money (MVND) to deposit each year", min_value=0.0)
# total_years = st.number_input("Total year", min_value=1)
# interest_rate = st.number_input("Interst rate", min_value=0.0)
# year_start_withdraw = int(st.number_input("Year to start withdrawn money"))
# withdrawal_year_23 = st.number_input("Money withdrawn at first year", min_value=0.0)
# withdrawal_increase = st.number_input("Withdrawal Increase per year", min_value=0.0)
# # Calculate future values
# future_values = [0] * total_years

# for year in range(1, year_start_withdraw):
#     P = future_values[year - 1] * (1 + interest_rate) + p_value
#     future_values[year] = P

# for year in range(year_start_withdraw, total_years):
#     withdrawal = withdrawal_year_23 * (1 + withdrawal_increase) ** (year - year_start_withdraw)
#     future_values[year] = future_values[year - 1] * (1 + interest_rate) - withdrawal

# # Display results
# st.write("Future value:", future_values[-1], "MVND")

# # Plot future values
# st.line_chart(future_values)
# st.xlabel("P")
# st.ylabel("future value (MVND)")

def ordinal(n: int):
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
    return str(n) + suffix

# Expected value function
def calculate_expected_return(principal, rate, tenor, tenor_unit, interest_type, start_date):
    """
    Calculates the expected return based on the type of interest accrual.

    Args:
    principal (float): Initial deposit amount.
    rate (float): Annual interest rate (percentage).
    tenor (float): Duration of the investment.
    tenor_unit (str): Unit of tenor ('days', 'months', 'years').
    interest_type (str): Type of interest ('simple', 'compounded_daily', 'compounded_monthly',
                         'compounded_quarterly', 'compounded_semi_yearly', 'compounded_yearly', 'continuous').
    start_date (str): Start date of the investment in 'YYYY-MM-DD' format.

    Returns:
    float: Expected return at the end of the investment period.
    """
    rate_decimal = rate / 100.0
    start_date = datetime.strptime(start_date, "%Y-%m-%d")

    # Convert tenor to years based on tenor_unit
    if tenor_unit == 'days':
        tenor_years = tenor / 365.0
    elif tenor_unit == 'months':
        tenor_years = tenor / 12.0
    elif tenor_unit == 'years':
        tenor_years = tenor

    # Calculate future value based on interest type
    if "compounded" in interest_type:
        compounding_periods = {'compounded_daily': 365, 'compounded_monthly': 12, 'compounded_quarterly': 4,
                               'compounded_semi_yearly': 2, 'compounded_yearly': 1}.get(interest_type, 1)
        future_value = principal * (1 + rate_decimal / compounding_periods) ** (compounding_periods * tenor_years)
    elif interest_type == "simple":
        future_value = principal * (1 + rate_decimal * tenor_years)
    elif interest_type == "continuous":
        future_value = principal * 2.718282**(rate_decimal * tenor_years)

    return future_value
# Daily tracking function
def investment_value_on_date(principal, rate, tenor, tenor_unit, interest_type, start_date, value_date=None):
    """
    Calculates the value of an investment on a specific date.

    Args:
    - principal (float): The initial amount of money invested.
    - rate (float): The annual interest rate as a percentage.
    - tenor (float): The duration of the investment.
    - tenor_unit (str): The unit of duration ('days', 'months', 'years').
    - interest_type (str): The type of interest calculation ('simple', 'compounded_daily', 'compounded_monthly',
                         'compounded_quarterly', 'compounded_semi_yearly', 'compounded_yearly', 'continuous').
    - start_date (str): The start date of the investment in 'YYYY-MM-DD' format.
    - value_date (str): Optional; the date to calculate the investment value for, in 'YYYY-MM-DD' format.
                       If not provided, uses today's date.

    Returns:
    - float: The value of the investment on the specified date.
    """
    rate_decimal = rate / 100.0
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if value_date:
        value_date = datetime.strptime(value_date, "%Y-%m-%d")
    else:
        value_date = datetime.today()

    # Calculating the total days of investment based on tenor_unit
    if tenor_unit == 'days':
        days_total = int(tenor)
    elif tenor_unit == 'months':
        days_total = int(tenor * 30)  # Rough approximation
    elif tenor_unit == 'years':
        days_total = int(tenor * 365)  # Ignoring leap years for simplicity

    end_date = start_date + timedelta(days=days_total)

    # Ensure value_date is within the investment period
    # if not start_date <= value_date <= end_date:
    #     raise ValueError("The specified value date is outside the investment period.")

    days_since_start = (value_date - start_date).days
    years_since_start = days_since_start / 365.0

    if interest_type == 'simple':
        value = principal * (1 + rate_decimal * years_since_start)
    elif interest_type == 'continuous':
        value = principal * 2.718282**(rate_decimal * years_since_start)
    else:
        compounding_periods = {
            'compounded_daily': 365,
            'compounded_monthly': 12,
            'compounded_quarterly': 4,
            'compounded_semi_yearly': 2,
            'compounded_yearly': 1
        }
        periods_since_start = int(compounding_periods[interest_type] * years_since_start)
        value = principal * (1 + rate_decimal / compounding_periods[interest_type]) ** periods_since_start

    return value
def intro():
    import streamlit as st
    st.write("# Welcome to Our Demo! 👋")
    st.sidebar.success("Select a funciton above.")

    st.markdown(
        """
        Streamlit is an open-source app framework built specifically for
        Machine Learning and Data Science projects.

        **👈 Select a demo from the dropdown on the left** to see some examples
        of what Streamlit can do!

        ### Want to learn more?

        - Check out [streamlit.io](https://streamlit.io)
        - Jump into our [documentation](https://docs.streamlit.io)
        - Ask a question in our [community
          forums](https://discuss.streamlit.io)

        ### See more complex demos

        - Use a neural net to [analyze the Udacity Self-driving Car Image
          Dataset](https://github.com/streamlit/demo-self-driving)
        - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
    """
    )

def saving_management():
    import streamlit as st
    import pandas as pd
    import pydeck as pdk

    from urllib.error import URLError

    st.markdown(f"# {list(page_names_to_funcs.keys())[1]}")
    #Saving Management
    # st.title('Saving Management')
    # Data Frame for Saving Management
    data = []
    more_profiles = True
    profile_nums = st.number_input("How many saving profiles do you have?", min_value=0)
    for i in range(profile_nums):
        st.subheader("The " + str(ordinal(i+1)) + " saving profile information input:" )
        # Input
        principal = float(st.number_input("Enter the initial deposit amount (VND): ", min_value=0.0, key = i))
        rate = float(st.number_input("Enter the interest rate (percentage per year):", min_value=0.0, key = i+0.5 ))
        tenor = float(st.number_input("Enter the tenor: ", min_value=0.0, key = i+0.75))
        tu_option = st.selectbox(
        "Select the unit of tenor (days, months, years):",
        ("days", "months", "years"), key = i+0.25)
        # st.write("You selected:", option)
        it_option = st.selectbox(
        "Select the unit of tenor (days, months, years):",
        ('compounded_daily', 'compounded_monthly', 'compounded_quarterly', 'compounded_semi_yearly', 'compounded_yearly'), key = i+0.55)
        tenor_unit = tu_option
        interest_type = it_option
        d = '2023-05-05'
        d = st.date_input("Enter the start date (YYYY-MM-DD)", key = i+0.35)
        st.write("Your start is:", d)
        start_date = d.strftime("%Y-%m-%d")
        
        # Output
        expected_return = calculate_expected_return(principal, rate, tenor, tenor_unit, interest_type, start_date)
        today_value = investment_value_on_date(principal, rate, tenor, tenor_unit, interest_type, start_date)

        data.append({
            "Principal": principal,
            "Rate": rate,
            "Tenor": tenor,
            "Tenor Unit": tenor_unit,
            "Interest Type": interest_type,
            "Start Date": start_date,
            "Expected Return": expected_return,
            "Today's Value": today_value
        })
        
        # if st.button("Yes"):
        #     more_profiles = 'yes'
        #     # st.write("Input another profile")
        # if st.button("No"):
        #     more_profiles = 'no'
            # st.write("Goodbye")
        # more = input("Add another profile? (yes/no): ")
        # more_profiles = more.lower() == 'yes'
    if st.button("Start calculate", type = "primary"):
        df = pd.DataFrame(data)
        st.write("Expected Return:", expected_return)
        st.write("Today Value:", today_value)
        # Calculate total sums
        total_expected_return = df["Expected Return"].sum()
        total_today_value = df["Today's Value"].sum()

        st.write("Total Expected Return:", total_expected_return)
        st.write("Total Today's Value:", total_today_value)
        # print(df.head())
        st.dataframe(df) 
        return 
    # Create DataFrame
   

# Saving planning strategy
def saving_planning_strategy():
    st.title('Saving Planning')
    data = []
    more_profiles = True
    profile_nums = st.number_input("How many saving profiles do you plan to have?", min_value=0)
    for i in range(profile_nums):
        st.subheader("The " + str(ordinal(i+1)) + " planning saving profile information input:" )
        # Input
        principal = float(st.number_input("Enter the initial deposit amount (VND): ", min_value=0.0, key = i))
        rate = float(st.number_input("Enter the interest rate (percentage per year): ", min_value=0.0, key = i+0.1))
        tenor = float(st.number_input("Enter the tenor: ", min_value=0.0, key = i+0.2))
        tu_option = st.selectbox(
        "Select the unit of tenor (days, months, years):",
        ("days", "months", "years"), key = i + 0.3)
        tenor_unit = tu_option
        it_option = st.selectbox(
        "Select the unit of tenor (days, months, years):",
        ('compounded_daily', 'compounded_monthly', 'compounded_quarterly', 'compounded_semi_yearly', 'compounded_yearly'), key = i + 0.5)
        interest_type = it_option
        d = '2023-05-05'
        d = st.date_input("Enter the start date (YYYY-MM-DD)", key = i + 0.4)
        st.write("Your start is:", d)
        start_date = d.strftime("%Y-%m-%d")

        # Output
        expected_return = calculate_expected_return(principal, rate, tenor, tenor_unit, interest_type, start_date)
        st.write(f"Expected Return for {principal} VND at {rate}% for {tenor} {tenor_unit} ({interest_type}): {expected_return:.2f} VND")

        # Store in a list for DataFrame
        data.append({
            "Principal": principal,
            "Rate": rate,
            "Tenor": tenor,
            "Tenor Unit": tenor_unit,
            "Interest Type": interest_type,
            "Start Date": start_date,
            "Expected Return": expected_return
        })

        # st.write("Add another profile")
        # columns = st.columns([1,1,1,1,1,1,1,1])
        # with columns[0]:
        #     if st.button("Yes"):
        #         more_profiles = 'yes'
        # with columns[1]:
        #     if st.button("No"):
        #         more_profiles = 'no'

    # Create DataFrame
    # df = pd.DataFrame(data)
    if st.button("Start calculate", type = "primary"):
         # Main function call
        df = pd.DataFrame(data)
        st.dataframe(df)  # Print all entries

        # Calculate total expected return
        total_expected_return = df["Expected Return"].sum()
        st.write("Total Expected Return across all accounts:", total_expected_return)
    
   

#Stock managemenet:
def calculate_investment_details(tickers, initial_prices, quantities, purchase_dates):
    """
    Calculate the investment details using tickers, initial prices, quantities, and purchase dates.

    Args:
    tickers (list of str): Stock tickers/codes.
    initial_prices (list of float): Initial prices of the stocks when purchased.
    quantities (list of int): Number of shares of each stock purchased.
    purchase_dates (list of str): Purchase dates of each stock.

    Returns:
    dict: Details of the investment including total initial investment, total present value, and individual returns.
          Also provides visualizations for stock price history and volatility.
    """
    current_prices = fetch_current_prices(tickers)
    investments = []
    total_initial_investment = 0
    total_present_value = 0
    total_return = 0

    for ticker, initial_price, quantity, purchase_date in zip(tickers, initial_prices, quantities, purchase_dates):
        current_price = current_prices[ticker]
        initial_investment = initial_price * quantity
        present_value = current_price * quantity
        individual_return = present_value - initial_investment
        roi = (individual_return / initial_investment) * 100

        total_initial_investment += initial_investment
        total_present_value += present_value
        total_return += individual_return

        investments.append({
            "Ticker": ticker,
            "Initial Price": initial_price,
            "Quantity": quantity,
            "Purchase Date": purchase_date,
            "Current Price": current_price,
            "Initial Investment": initial_investment,
            "Present Value": present_value,
            "Individual Return": individual_return,
            "ROI (%)": roi
        })

    # Visualization of price history and volatility
    fig, axs = plt.subplots(len(tickers), 2, figsize=(15, 5 * len(tickers)))
    for idx, investment in enumerate(investments):
        ticker = investment["Ticker"]
        # Historical price chart
        historical_prices = fetch_historical_prices(ticker, investment["Purchase Date"])
        axs[idx][0].plot(historical_prices.index, historical_prices.values)
        axs[idx][0].set_title(f'Price History of {ticker}')
        axs[idx][0].set_xlabel('Date')
        axs[idx][0].set_ylabel('Price (USD)')

        # Volatility chart
        daily_returns = historical_prices.pct_change()
        axs[idx][1].bar(daily_returns.index, daily_returns.values)
        axs[idx][1].set_title(f'Volatility of {ticker}')
        axs[idx][1].set_xlabel('Date')
        axs[idx][1].set_ylabel('Daily Returns')

    plt.tight_layout()
    plt.show()

    return {
        "Total Initial Investment": total_initial_investment,
        "Total Present Value": total_present_value,
        "Total Return": total_return,
        "Investment Details": investments
    }
       
def fetch_current_prices(tickers):
    """
    Fetches the current prices for the given stock tickers using Yahoo Finance.
    """
    prices = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        # Fetch the last day's data to get the most recent closing price
        try:
            hist = stock.history(period="1d")
            # We use 'Close' of the last available day for the current price
            prices[ticker] = hist['Close'].iloc[-1] if not hist.empty else None
        except Exception as e:
            st.write(f"Failed to fetch data for {ticker}: {str(e)}")
            # If there's an issue with fetching, prompt user for manual input
            prices[ticker] = float(input(f"Enter the current price for {ticker} manually (USD): "))
    return prices
def fetch_historical_prices(ticker, start_date):
    """
    Fetches historical prices for a single stock ticker from the specified start date to the present, using Yahoo Finance.
    """
    stock = yf.Ticker(ticker)
    try:
        hist = stock.history(start=start_date)
        return hist['Close']
    except Exception as e:
        print(f"Failed to fetch historical data for {ticker}: {str(e)}")
        return pd.Series()
# Stock Analysis function
def calculate_investment_details(tickers, initial_prices, quantities, purchase_dates):
    """
    Calculate the investment details using tickers, initial prices, and quantities.

    Args:
    tickers (list of str): List of stock tickers.
    initial_prices (list of float): Initial prices of the stocks when purchased.
    quantities (list of int): Quantities of each stock purchased.

    Returns:
    dict: Details of the investment.
    """
    current_prices = fetch_current_prices(tickers)
    investments = []
    total_initial_investment = 0
    total_present_value = 0
    total_return = 0

    for ticker, initial_price, quantity, purchase_date in zip(tickers, initial_prices, quantities, purchase_dates):
        current_price = current_prices[ticker]
        initial_investment = initial_price * quantity
        present_value = current_price * quantity
        individual_return = present_value - initial_investment
        roi = (individual_return / initial_investment) * 100

        total_initial_investment += initial_investment
        total_present_value += present_value
        total_return += individual_return

        investments.append({
            "Ticker": ticker,
            "Initial Price": initial_price,
            "Quantity": quantity,
            "Purchase Date": purchase_date,
            "Current Price": current_price,
            "Initial Investment": initial_investment,
            "Present Value": present_value,
            "Individual Return": individual_return,
            "ROI (%)": roi
        })

    # Visualization of price history and volatility
    fig, axs = plt.subplots(len(tickers), 2, figsize=(15, 5 * len(tickers)))

    # Check the shape of axs and adjust indexing accordingly
    if len(tickers) == 1:
        axs = np.array([axs])

    for idx, investment in enumerate(investments):
        ticker = investment["Ticker"]
        # Historical price chart
        historical_prices = fetch_historical_prices(ticker, investment["Purchase Date"])
        axs[idx][0].plot(historical_prices.index, historical_prices.values)
        axs[idx][0].set_title(f'Price History of {ticker}')
        axs[idx][0].set_xlabel('Date')
        axs[idx][0].set_ylabel('Price (USD)')

        # Volatility chart
        daily_returns = historical_prices.pct_change()
        axs[idx][1].bar(daily_returns.index, daily_returns.values)
        axs[idx][1].set_title(f'Volatility of {ticker}')
        axs[idx][1].set_xlabel('Date')
        axs[idx][1].set_ylabel('Daily Returns')

    plt.tight_layout()
    st.pyplot(fig)




    # # Volatility plot
    # # Create a new figure and set its size
    # fig, ax = plt.subplots(figsize=(10, 6))

    # # Plot the data
    # ax.plot(individual_returns, 'o-')

    # # Set the title and labels
    # ax.set_title('Volatility Plot for Individual Stocks')
    # ax.set_xlabel('Stocks')
    # ax.set_ylabel('Returns')

    # # Set the xticks
    # ax.set_xticks(range(len(tickers)))
    # ax.set_xticklabels(tickers)

    # # Enable grid
    # ax.grid(True)

    # # Display the plot in Streamlit
    # st.pyplot(fig)

    return {
        "Total Initial Investment": total_initial_investment,
        "Total Present Value": total_present_value,
        "Total Return": total_return,
        "Investment Details": investments
    }
# Stock Portfolio Management
def stock_portfolio_management():
    st.title("Stock Portfolio Management")
    data = []
    tickers = []
    initial_prices = []
    quantities = []
    purchase_dates = []
    more_stocks = True
    stock_nums = st.number_input("How many stock are you holding?", min_value=0)
    for i in range(stock_nums):
        st.subheader("The " + str(ordinal(i+1)) + " stock information input:" )
        # Input
        ticker = st.text_input("Enter the stock " + str(i) +" ticker: ", key = i)
        tickers.append(ticker)
        quantity = int(st.number_input("Enter the quantity of shares: ", key = i + 0.5))
        quantities.append(quantity)
        initial_price = float(st.number_input("Enter the initial price per share (VND): ", key = i+0.75))
        initial_prices.append(initial_price)
        d = '2023-05-05'
        d = st.date_input("Enter the purchase date (YYYY-MM-DD)", key = i + 0.4)
        st.write("Your purchase date is:", d)
        purchase_date = d.strftime("%Y-%m-%d")
        purchase_dates.append(purchase_date)

        # Store each stock's data
        data.append({
            "Ticker": ticker,
            "Quantity": quantity,
            "Initial Price": initial_price
        })
    if st.button("Start calculate", type = "primary"):
        results = calculate_investment_details(tickers, initial_prices, quantities, purchase_dates)
        investment_details = results["Investment Details"]
        # Display results
        st.write("Investment Details:", investment_details)
        st.write("Total Initial Investment:", results["Total Initial Investment"])
        st.write("Total Present Value:", results["Total Present Value"])
        st.write("Total Return:", results["Total Return"])
        df = pd.DataFrame(data)
        st.dataframe(df) 

def stock_planning():
    return
page_names_to_funcs = {
    "Tutorial": intro,
    "Saving Management": saving_management,
    "Saving planning strategy": saving_planning_strategy,
    "Stock Portfolio Management": stock_portfolio_management,
    "Stock Planning": stock_planning

    # "DataFrame Demo": data_frame_demo
}

demo_name = st.sidebar.selectbox("Choose a function", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
