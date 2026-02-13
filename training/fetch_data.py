import requests
import pandas as pd
import json

URL = "https://container-chaos-b2c6bde72854.herokuapp.com/api/energy/production/outputs"

def fetch():
    res = requests.get(URL)
    print(res)
    data = res.json()

    df = pd.DataFrame(data)
    df.to_csv("data/raw.csv", index=False)

    print("Saved data/raw.csv")


def fetch_and_send():

    print("Loading drift_log.csv...")

    # Load drift data
    try:
        df = pd.read_csv("data/drift_log.csv")
    except Exception as e:
        print("Error reading CSV:", e)
        return


    if df.empty:
        print("drift_log.csv is empty!")
        return


    # Convert to JSON array
    data = df.to_dict(orient="records")


    print(f"Total Records to Send: {len(data)}")

    print("\nSample Data:")
    print(json.dumps(data[:2], indent=2))


    # Send POST request
    try:
        res = requests.post(
            URL,
            json=data,   # 👈 Automatically converts to JSON
            timeout=30
        )
    except Exception as e:
        print("Request Error:", e)
        return


    # Response info
    print("\nStatus Code:", res.status_code)

    if res.text:
        print("Response Body:")
        print(res.text)
    else:
        print("Empty Response")


    # Try parsing JSON
    try:
        response_json = res.json()
        print("\nResponse JSON:")
        print(json.dumps(response_json, indent=2))
    except:
        print("\nResponse is not JSON")


if __name__ == "__main__":
    fetch_and_send()
