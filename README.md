# Housing Price Prediction - Linear Regression

A Streamlit web application for predicting housing prices using Linear Regression with real Kaggle data.

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone/Navigate to the project directory:**
   ```bash
   cd /Users/toshitha/Desktop/week-18
   ```

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Kaggle API credentials (if not already configured):**
   - Visit https://www.kaggle.com/settings/account
   - Click "Create New API Token" to download `kaggle.json`
   - Place the file in `~/.kaggle/kaggle.json`
   - Run: `chmod 600 ~/.kaggle/kaggle.json` (on macOS/Linux)

### Running the Application

```bash
streamlit run streamlit_app.py
```

The app will open in your default browser at `http://localhost:8501`

## 📊 Features

### 🏠 Home Page
- Overview of the application
- Dataset statistics and information
- Quick stats about the housing market data

### 🤖 Model Training
- View model performance metrics
- Training and testing R² scores
- RMSE and MAE metrics
- Top feature coefficients visualization

### 💰 Predictions
- Interactive input form for property features
- Real-time price predictions
- Input feature display

### 📈 Model Analysis
- **Predictions Tab:** Actual vs Predicted scatter plots for train/test sets
- **Residuals Tab:** Distribution analysis of prediction errors
- **Feature Importance Tab:** Top features impacting price (coefficient magnitude)
- **Data Distribution Tab:** Correlation heatmap and price distribution

### ℹ️ About
- Model information and performance metrics
- Technologies used
- Feature overview

## 📁 Files

- `streamlit_app.py` - Main Streamlit application
- `linear.ipynb` - Jupyter notebook with full model development
- `requirements.txt` - Python dependencies
- `README.md` - This file

## 🔧 Project Structure

```
/Users/toshitha/Desktop/week-18/
├── streamlit_app.py      # Main Streamlit app
├── linear.ipynb          # Jupyter notebook
├── requirements.txt      # Dependencies
└── README.md            # Documentation
```

## 📊 Dataset Information

- **Source:** Kaggle (yasserh/housing-prices-dataset)
- **Records:** 545 properties
- **Features:** 13 (area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus, price)
- **Target:** Price (house price in currency units)

## 🎯 Model Performance

| Metric | Value |
|--------|-------|
| Training R² | 0.6859 |
| Testing R² | 0.6529 |
| RMSE | $1,324,506.96 |
| MAE | $970,043.40 |

## 🔝 Top Predictive Features

1. **Bathrooms** - Strong positive impact on price
2. **Air Conditioning** - Increases price significantly
3. **Hot Water Heating** - Adds substantial value
4. **Preferred Area** - Premium location impact
5. **Parking** - Important amenity

## 🌐 Deployment Options

### Option 1: Streamlit Cloud (Free & Recommended)
1. Push your code to GitHub
2. Go to https://streamlit.io/cloud
3. Connect your GitHub repository
4. Deploy with one click

### Option 2: Heroku
1. Create a `Procfile`:
   ```
   web: streamlit run streamlit_app.py
   ```
2. Deploy using Heroku CLI

### Option 3: AWS, Azure, or Google Cloud
- Create a virtual machine or use container services (Docker)
- Install dependencies and run `streamlit run streamlit_app.py`

### Option 4: Local Network
```bash
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

## 🐛 Troubleshooting

### Issue: "No module named 'kagglehub'"
**Solution:** Run `pip install kagglehub`

### Issue: "Kaggle API error"
**Solution:** Ensure kaggle.json is in ~/.kaggle/ with proper permissions

### Issue: "Port 8501 already in use"
**Solution:** Run on a different port:
```bash
streamlit run streamlit_app.py --server.port 8502
```

## 📝 Notes

- The model is trained fresh each time the app loads (cached for performance)
- All predictions are based on the Linear Regression model trained on Kaggle housing data
- Feature inputs are constrained to the data ranges observed in the training set
- For better predictions, consider collecting more recent property data

## 📚 References

- [Streamlit Documentation](https://docs.streamlit.io)
- [Scikit-learn Linear Regression](https://scikit-learn.org/stable/modules/linear_model.html)
- [Kaggle Housing Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)

## 💡 Future Enhancements

- Add model persistence (save trained model)
- Implement multiple regression models for comparison
- Add data upload feature for batch predictions
- Include cross-validation analysis
- Add confidence intervals to predictions
- Implement advanced feature engineering

## 📧 Support

For issues or questions, refer to the documentation files or check the Streamlit community forums.

---
**Created:** May 2024 | **Last Updated:** May 2024
