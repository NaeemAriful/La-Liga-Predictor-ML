# ⚽ La Liga Match Predictor (2000–2025)

This project uses **26 years of historical La Liga data** to predict football match outcomes (Win, Loss, or Draw). 

### Key Features
* **Elo Ratings:** Dynamic team strength tracking over two decades.
* **EWMA Form:** Exponentially Weighted Moving Averages to prioritize recent team performance.
* **Accuracy:** Reaches a solid **52.33%** predictive accuracy in a high-variance sport.

###  How to Use
1. Clone the repo: `git clone https://github.com/NaeemAriful/La-Liga-Predictor-ML.git`
2. Ensure you have the raw CSVs in `data/raw/`.
3. Run the main script: `python scripts/laliga.py`

###  Sample Prediction
For **Girona vs Getafe**, the model predicts:
- **Girona Win:** 48.55%
- **Draw:** 26.43%
- **Getafe Win:** 25.01%
