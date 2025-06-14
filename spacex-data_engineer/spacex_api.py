import pandas as pd
import requests

def main():
    response = requests.get('https://api.spacexdata.com/v4/launches')
    data = response.json()
    df = pd.DataFrame(data=data)
    print(df.head())

if __name__ == "__main__":
    main()