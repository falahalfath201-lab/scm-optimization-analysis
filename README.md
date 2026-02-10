# Dataset Information

This project uses the **DataCo Smart Supply Chain Dataset** from Kaggle.

## How to Download

### Option 1: Manual Download (Easiest)

1. Visit the dataset page: [DataCo Smart Supply Chain Dataset](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis)
2. Click the **Download** button (you may need to sign in to Kaggle)
3. Extract the downloaded ZIP file
4. Place `DataCoSupplyChainDataset.csv` in this folder (`data/raw/`)

### Option 2: Using Kaggle API

If you have Kaggle API credentials configured:

```bash
# Install Kaggle API
pip install kaggle

# Download dataset
kaggle datasets download -d shashwatwork/dataco-smart-supply-chain-for-big-data-analysis

# Extract to this folder
unzip dataco-smart-supply-chain-for-big-data-analysis.zip -d data/raw/
```

## Expected File

After downloading, you should have:
```
data/raw/DataCoSupplyChainDataset.csv
```

## Dataset Details

- **Source:** Kaggle
- **Size:** ~100MB
- **Records:** 180,519 rows
- **Features:** 53 columns including order details, shipping information, customer data, and product information

## Note

The dataset is **not included** in this repository due to:
- Size constraints (GitHub file size limits)
- Licensing (the dataset is available on Kaggle under their terms)
- Best practice (keeping code and data separate)

Please download it following the instructions above before running the analyses.
