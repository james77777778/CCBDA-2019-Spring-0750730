### Usage
1. Train Model
    ```python
    python download_stock.py
    ```
    Download the stock infos from Yahoo Finance see **2019-XX-XX_stocks.json** and **stocks.json** two identical files.
    ```python
    python train.py
    ```
    Train 20 LSTM models and save their best state_dicts at **models/**.
2. Predict
    ```python
    python predict.py
    ```
    Use the best state_dicts to predict the next day stock trend.
    Save the results in **results.txt**.
