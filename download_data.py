import yfinance as yf
import os

START = "2019-01-01"
END   = "2023-12-31"

os.makedirs("data", exist_ok=True)

def download_and_save(ticker, filename):
    try:
        print(f"Downloading {ticker}...")

        data = yf.download(ticker, start=START, end=END)

        if data.empty:
            print(f"❌ Failed: {ticker}")
            return

        data["Close"].to_csv(f"data/{filename}.csv")
        print(f"✅ Saved: data/{filename}.csv")

    except Exception as e:
        print(f"❌ Error with {ticker}: {e}")


def main():
    # Companies
    download_and_save("TCS.NS", "tcs")
    download_and_save("BAJFINANCE.NS", "bajaj")
    download_and_save("VEDL.NS", "vedanta")

    # Market + Economy
    download_and_save("^BSESN", "sensex")
    download_and_save("INR=X", "usd_inr")


if __name__ == "__main__":
    main()