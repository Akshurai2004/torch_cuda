# CUDA Error Resolution Analysis Dashboard

Interactive Streamlit dashboard for analyzing CUDA-related question resolution times on the PyTorch forum.

## Features

- **Data Overview**: View statistics and visualizations comparing CUDA vs non-CUDA questions
- **Resolution Time Prediction**: Use ML models to predict how long a question will take to resolve
- **Causal Analysis Results**: Explore causal inference findings on the impact of CUDA-related questions

## Requirements

The dashboard requires the following data files:

1. **Forum Data** (optional for full functionality):
   - `data/processed/forum_data.csv`
   - Contains scraped forum questions with features

2. **Causal Analysis Results** (required for Causal Analysis page):
   - `data/processed/causal_results.json`
   - JSON format with structure:
   ```json
   [
     {
       "method": "Method Name",
       "ate": 4.2,
       "ci_lower": 3.1,
       "ci_upper": 5.3
     }
   ]
   ```

3. **Prediction Model** (optional):
   - `data/models/multiple_regression_model.pkl`
   - Trained scikit-learn model for resolution time prediction

## Running the Dashboard

1. Install dependencies:
   ```bash
   pip install -e .
   ```

2. Ensure your `causal_results.json` file is placed in:
   ```
   data/processed/causal_results.json
   ```

3. Run the dashboard:
   ```bash
   streamlit run dashboard/app.py
   ```

4. Open your browser to the URL shown (typically http://localhost:8501)

## Data File Format

### causal_results.json

The causal results file should contain an array of objects with the following fields:

- `method` (string): Name of the causal inference method
- `ate` (number): Average Treatment Effect in hours
- `ci_lower` (number): Lower bound of 95% confidence interval
- `ci_upper` (number): Upper bound of 95% confidence interval

Example:
```json
[
  {
    "method": "Propensity Score Matching",
    "ate": 4.3,
    "ci_lower": 3.2,
    "ci_upper": 5.4
  },
  {
    "method": "Double ML",
    "ate": 4.2,
    "ci_lower": 3.1,
    "ci_upper": 5.3
  }
]
```

## Troubleshooting

**"Causal analysis results not found" warning:**
- Ensure `data/processed/causal_results.json` exists
- Check that the JSON file is valid and follows the expected format
- Run your causal analysis script to generate the file

**"Data file not found" error:**
- You need `data/processed/forum_data.csv` for the dashboard to work
- Run the data collection and processing scripts first

**Model not available:**
- The prediction feature requires a trained model at `data/models/multiple_regression_model.pkl`
- This is optional - other pages will still work without it
