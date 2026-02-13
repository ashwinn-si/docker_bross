import requests
import pandas as pd

URL = "https://container-chaos-b2c6bde72854.herokuapp.com/api/energy/train-test"

def fetch():
    res = requests.get(URL)
    print(res)
    data = res.json()

    df = pd.DataFrame(data)
    df.to_csv("data/raw.csv", index=False)

    print("Saved data/raw.csv")

if __name__ == "__main__":
    fetch()
