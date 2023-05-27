# UFC-Betting-Model
Streamlit dashboard that uses public UFC fight data and a statistical model to generate fight result predictions

Main functions of app:
1. Data is scraped from http://ufcstats.com/statistics/events/completed, cleaned, stored in SQLite DB. 
2. Calculation of advanced stats (i.e, rolling average of significant strikes).
3. Random forests model constructed using stats from #2. (WIN model, KO model, SUBMISSION model)
4. Streamlit GUI that allows user to pick fighters from roster and compare model's odds vs online betting site odds. Program also provides a recommended bet amount depending on the difference between model prediction and online betting site prediction.
5. Visual tracking of results + bet history via Streamlit GUI.

Python, Numpy, SQLlite DB, Streamlit
