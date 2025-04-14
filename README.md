## Install dependencies

```bash
pip install pandas xgboost
```

Dependencies:
- [Pandas](https://pandas.pydata.org/): 2.2.4
- [XGBoost](https://xgboost.readthedocs.io/en/release_3.0.0/): 2.1.4

## Running sales revenue forecaster

```bash
python forecast.py --date --venue_id --concept --city --country
```

- **date** - The operational date for the sales revenue.
- **venue_id** - Unique identifier for the venue.
- **concept** - The type or category of the venue.
- **city** - The city where the venue is located.
- **country** - The country where the venue is located.
