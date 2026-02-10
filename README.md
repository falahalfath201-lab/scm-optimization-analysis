# Supply Chain Optimization Analysis

A comprehensive supply chain optimization system that provides multiple analyses including cost optimization, route optimization, delivery prediction, risk analysis, and warehouse location optimization.

## Features

- **Cost Optimization**: Optimize shipping costs across the supply chain
- **Route Optimization**: Find optimal delivery routes using advanced algorithms
- **Delivery Prediction**: Predict delivery times using machine learning models
- **Risk Analysis**: Analyze and predict late delivery risks
- **Warehouse Location**: Optimize warehouse locations for better distribution

## Project Structure

```
scm-optimization-analysis/
├── src/                  # Source code
│   ├── analysis/        # Analysis modules
│   ├── models/          # Machine learning models
│   ├── pipelines/       # Analysis pipelines
│   └── utils/           # Utility functions
├── data/                # Data folder
│   └── README.md        # Dataset installation instructions
├── outputs/             # Generated reports and figures
├── tests/               # Test files
├── main.py              # Main entry point
└── requirements.txt     # Python dependencies
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/falahalfath201-lab/scm-optimization-analysis.git
cd scm-optimization-analysis
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Dataset

The project requires the DataCo Smart Supply Chain Dataset. Follow the instructions in [`data/README.md`](data/README.md) to download and setup the dataset.

## Usage

### Running the System

Execute the main script to start the interactive menu:

```bash
python main.py
```

### Execution Modes

The system offers three execution modes:

#### Mode 1: Single Analysis (Fast)
Run one specific analysis independently. Best for development and quick insights.

```
Available analyses:
  1. Cost Optimization
  2. Route Optimization
  3. Delivery Prediction
  4. Risk Analysis
  5. Warehouse Location
```

#### Mode 2: Selected Analyses (Medium)
Run multiple selected analyses with shared data loading. More efficient when running multiple analyses.

#### Mode 3: All Analyses (Comprehensive)
Run the complete optimization suite. Generates comprehensive reports with all insights.

### Output

Results are saved in the `outputs/` directory:
- **Reports**: `outputs/reports/` - Analysis reports and summaries
- **Figures**: `outputs/figures/` - Visualizations and charts
- **Data**: `data/` - Result CSV files (e.g., cost_optimization_results_*.csv)

## Requirements

- Python 3.8 or higher
- See `requirements.txt` for complete list of dependencies

Key libraries:
- pandas, numpy, scipy for data processing
- scikit-learn for machine learning
- ortools, pulp for optimization
- torch, stable-baselines3 for reinforcement learning
- matplotlib, seaborn, plotly for visualization
- geopy, geopandas for geographical analysis

## Dataset

This project uses the **DataCo Smart Supply Chain Dataset** from Kaggle. The dataset contains 180,519 rows and 53 columns with information about:
- Order details
- Shipping information
- Customer data
- Product information

For dataset download and setup instructions, see [`data/README.md`](data/README.md).

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
