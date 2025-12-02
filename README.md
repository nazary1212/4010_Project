# Construction Permit Forecasting - 4010 Deep Learning for Time Series Forecasting Project 

## Project overview

This project builds time-series forecasting models for U.S. residential construction permits using a historical dataset of monthly permit counts and valuations by state and unit type. It compares several neural and statistical forecasting models, applies hyperparameter optimization, and produces 1‑year ahead forecasts by state for total permits given out as well as individiual permit types for 1 unit, 2 unit, 3-4 unit, and 5+ unit residencies 

## Data description

The dataset is stored in `construction_permits.csv` and contains monthly observations by U.S. state, including:  
- State name (`Name`).  
- Number of permits by unit size: 1 unit, 2 units, 3–4 units, 5+ units.  
- Date (full period string, numeric month, month name, year).  
- Valuations for each unit-size category.  

These fields are used to construct total-permit time series and permit-type-specific series for each state for model training and evaluation.

## Data preprocessing

The preprocessing section of the script:  
- Loads `construction_permits.csv`, inspects the head of the table, and identifies any columns with missing values (N/A).  
- Constructs relevant time indices and derived series (such as total permits per state) to prepare data for time-series modeling.  

This stage ensures the dataset is clean and correctly structured before generating baselines and advanced models.

## Total permits analysis

### Baseline prediction on full history

For total permits per state, the code first trains a baseline recurrent neural network (RNN) model on the full available time horizon. The goal is to quickly gauge how difficult each state is to forecast by computing root mean squared error (RMSE) across states and identifying states with relatively high or low prediction error.

### Baseline post‑2008 financial crisis

To better align with the regulatory environment on building permits handed out in the U.S relevant for post‑2025 forecasting, a second baseline RNN is trained on data starting from approximately January 1, 2009. This subset aims to reduce noise introduced by structural changes in permit regulations during and immediately after the 2008 financial crisis. 

## Model selection for total permits

The “Choosing a forecasting model” section compares several models on total permits: GRU, RNN, TCN, and SARIMA. Models are evaluated using a 10‑state subset that excludes clear outliers (top 3 highest and lowest RMSE states from the earlier baseline), and their performance is ranked by average RMSE.

At the end of this section, the models are ordered from best to worst based on these results, and the top-performing model is selected for hyperparameter optimization on a per‑state basis.

## Hyperparameter optimization (HPO) on total permits

In the “HPO on total permits” section, the script runs hyperparameter optimization on the best model identified above. This procedure searches for state‑specific hyperparameters so that each state receives a tailored model configuration rather than a single global setting, improving RMSE across the board.

## One‑year‑ahead forecasts (total permits)

The “1 YR FORECAST” section generates 12‑month‑ahead forecasts of total permits for every state using the optimized model configurations. For each state, the script produces a forecasted trajectory over the next year, which can be visualized as plots for interpretation and reporting.

## Permit type analysis

### Baseline models for permit types

The “PERMIT TYPES” section focuses on forecasting separate series for each permit type (for example, 1‑unit, 2‑unit, 3–4‑unit, 5+‑unit permits) following a workflow analogous to the total permit analysis. Baseline models using RNN, TCN, and GRU are trained and evaluated on the same 10‑state subset to compare performance by permit type.

### Selecting best model per permit type

After computing average RMSE across the 10 selected states, the script ranks the models for each permit type and identifies the first and second best performers. The best model per permit type is then chosen for further refinement and hyperparameter optimization, enabling different architectures for different permit categories if beneficial.

## Hyperparameter optimization for permit types

In the “HPO – Best model” section, hyperparameter optimization is applied to the selected models for each permit type. For example, TCN is optimized for 1‑unit permits, while GRU is optimized for the remaining permit types, reflecting earlier comparative results.

## Final interactive results

The “Final Results” section offers an interface where a user can specify a state and permit type to generate and visualize a 1‑year ahead forecast. For each chosen state–permit combination, the script plots the forecast along with a simple confidence interval constructed by shifting predictions by plus/minus one standard deviation of historical values for that specific state and permit type.

## How to run

- Ensure Python and the required scientific/ML libraries (for example, NumPy, pandas, TensorFlow/PyTorch, statsmodels, and an HPO library such as Optuna if used) are installed.  
- Place `construction_permits.csv` in the same directory as the main script and run the script from the command line or an IDE.  
- Follow any prompts or function arguments in the “Final Results” section to input a state and permit type and view the generated forecast plots.
