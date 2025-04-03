# King County House Price Prediction

This repository contains a Python script that uses linear regression to predict house prices based on the King County house dataset. It explores both simple linear regression with single features and multiple regression with a selected feature set.

## Dataset
- **Source**: `kc_house_data.csv` (King County house sales data)
- **Description**: Contains house sale prices and features like bedrooms, bathrooms, square footage, etc.
- **Size**: 21,613 examples, 21 columns (including `id` and `price`)
- **Note**: Dataset must be downloaded separately (e.g., from Kaggle: [House Sales in King County, USA](https://www.kaggle.com/harlfoxem/housesalesprediction))

## Features
- **Data Exploration**
  - Lists all columns and counts features/examples
- **Simple Linear Regression**
  - Models price using single features (e.g., bedrooms)
  - Compares R² scores across multiple features
  - Visualizes the best single feature vs. price
- **Multiple Linear Regression**
  - Uses 9 selected features for prediction
  - Reports training and testing R² scores

## Requirements
```bash
pip install pandas numpy sklearn matplotlib

## Usage
1. Place `kc_house_data.csv` in the working directory
2. Run the script:
```bash
python house_price_regression.py
```
3. View outputs:
   - Console: R² scores and feature comparisons
   - Plot: Scatterplot with regression line for the best single feature

## Script Structure
- **Data Loading**: Reads the CSV file
- **Simple Regression**: 
  - Tests `bedrooms` alone
  - Compares multiple single features
  - Plots the best feature
- **Multiple Regression**: Trains on a subset of 9 features
- **Evaluation**: Prints R² scores for all models

## Configuration
- **Features Tested**: 
  - Single: `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `waterfront`
  - Multiple: `bedrooms`, `bathrooms`, `sqft_living`, `sqft_above`, `floors`, `waterfront`, `view`, `sqft_basement`, `lat`
- **Test Size**: 20% (random_state=42)

## Output
- **Console**:
  ```
  All columns in dataset: ['id', 'date', 'price', ...]
  Number of features used for training: 19
  Number of examples in the dataset: 21613

  Simple Linear Regression (bedrooms):
  R² training = 0.095
  R² testing  = 0.099

  Feature: bedrooms, R² testing = 0.099
  Feature: sqft_living, R² testing = 0.492
  ...
  Best single feature is 'sqft_living' with R² testing = 0.492

  Multiple Regression Model:
  R² training = 0.614
  R² testing  = 0.621
  ```
- **Plot**: Scatterplot of the best feature (e.g., `sqft_living`) vs. price with regression line

## Example Output
![Sample Plot](scatterplot_example.png) *(Add this file manually if desired)*

## Notes
- Ensure `kc_house_data.csv` is present, or the script will fail
- `sqft_living` typically emerges as the best single predictor (R² ~0.49)
- Multiple regression improves performance (R² ~0.62)
- Random state is fixed (42) for reproducibility
- Customize `feature_set` or `features_multi` to experiment with other columns

## License
This project is open-source and available under the MIT License.
```

### Key Points
- **Overview**: Describes the purpose and dataset.
- **Features**: Highlights exploration, simple, and multiple regression components.
- **Requirements**: Lists necessary packages.
- **Usage**: Provides clear instructions and expected outputs.
- **Configuration**: Notes the features used and test split.
- **Output**: Details console logs and plot, with a placeholder for an example image.
- **Notes**: Includes practical tips and customization options.

This README is tailored to your script, offering a user-friendly guide while covering all essential aspects. Let me know if you'd like to adjust anything or add a specific example output!
