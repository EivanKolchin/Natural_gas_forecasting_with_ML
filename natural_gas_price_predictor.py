import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from sklearn.linear_model import LinearRegression


df = pd.read_csv("Nat_Gas.csv")
df["Date"] = pd.to_datetime(df["Dates"])
df = df.sort_values("Date").reset_index(drop=True)

first_date = df["Date"].min().date()
last_date  = df["Date"].max().date()
max_allowed_date = last_date + timedelta(days=365)

df["t_years"] = (df["Date"] - pd.Timestamp(first_date)).dt.days / 365.0

df["Month"] = df["Date"].dt.month
df["doy"]   = df["Date"].dt.dayofyear
df["sin_doy"] = np.sin(2 * np.pi * df["doy"] / 365.0)
df["cos_doy"] = np.cos(2 * np.pi * df["doy"] / 365.0)

X = df[["t_years", "sin_doy", "cos_doy", "Month"]]
y = df["Prices"].values


X_train = pd.get_dummies(X, columns=["Month"], drop_first=True)
model = LinearRegression()
model.fit(X_train, y)
historical_price_map = dict(zip(df["Date"].dt.date, df["Prices"]))


def estimate_price(target_date):
    if isinstance(target_date, str):
        target_date = datetime.strptime(target_date, "%Y-%m-%d").date()
    elif isinstance(target_date, datetime):
        target_date = target_date.date()
    elif not isinstance(target_date, date):
        raise TypeError(
            "target_date must be a 'YYYY-MM-DD' string, datetime.date, or datetime.datetime"
        )

    if target_date < first_date:
        raise ValueError(
            f"Date {target_date} is before first observation {first_date}"
        )
    if target_date > max_allowed_date:
        raise ValueError(
            f"Date {target_date} is more than 1 year after last observation {last_date}"
        )

    if target_date in historical_price_map:
        return float(historical_price_map[target_date])

    t_years = (pd.Timestamp(target_date) - pd.Timestamp(first_date)).days / 365.0
    month   = target_date.month
    doy     = target_date.timetuple().tm_yday

    sin_doy = np.sin(2 * np.pi * doy / 365.0)
    cos_doy = np.cos(2 * np.pi * doy / 365.0)

    X_new = pd.DataFrame({
        "t_years": [t_years],
        "sin_doy": [sin_doy],
        "cos_doy": [cos_doy],
        "Month":   [month]
    })

    X_new = pd.get_dummies(X_new, columns=["Month"], drop_first=True)
    X_new = X_new.reindex(columns=X_train.columns, fill_value=0)

    return float(model.predict(X_new)[0])


def build_estimated_series(start_date=None, end_date=None, freq="D"):
    if start_date is None:
        start_date = first_date
    if end_date is None:
        end_date = max_allowed_date

    dates = pd.date_range(start_date, end_date, freq=freq)
    prices = [estimate_price(d.date()) for d in dates]
    return pd.Series(prices, index=dates, name="EstimatedPrice")

while True:
    d = input('Enter date, in format: "XXXX-XX-XX"\n(Year-Month-Day)\n\n    > ')
    print(f"Estimated price on {d}: {estimate_price(d):.2f}")
