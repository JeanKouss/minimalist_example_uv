# Minimalist example of model integration with CHAP (uv version)

This document demonstrates a minimalist example of how to write a CHAP-compatible forecasting model using modern Python tooling. The example uses [uv](https://docs.astral.sh/uv/) for dependency management.

The model simply learns a linear regression from rainfall and temperature to disease cases in the same month, without considering any previous disease or climate data. It also assumes and works only with a single region. The model is not meant to accurately capture any interesting relations - the purpose is just to show how CHAP integration works in a simplest possible setting.

## Requirements

Before running this example, you need to have [uv](https://docs.astral.sh/uv/) installed on your system. If you don't have it, install it with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```


## Repository structure

```
.
├── MLproject           # CHAP integration configuration
├── train.py            # Training logic
├── predict.py          # Prediction logic
├── pyproject.toml      # Python dependencies
├── isolated_run.py     # Script for testing without CHAP
├── input/              # Sample training and forecast data
└── output/             # Generated models and predictions
```

### Key files

- **MLproject**: Defines how CHAP interacts with your model (entry points, parameters)
- **train.py**: Contains the training logic
- **predict.py**: Contains the prediction logic
- **pyproject.toml**: Lists your Python dependencies - uv uses this to create the virtual environment
- **isolated_run.py**: Allows testing your model standalone, without CHAP

No other setup is needed - `uv run` will automatically create the virtual environment and install dependencies on first use.

## Running the model without CHAP integration

Before getting a new model to work as part of CHAP, it can be useful to develop and debug it while running it directly on a small dataset from file. Sample data files are included in the `input/` directory.

To quickly test everything works, run:

```bash
python isolated_run.py
```

This will train the model and generate predictions using the sample data. You can also run the commands manually:

### Training the model

The file `train.py` contains the code to train a model. Run it with:

```bash
uv run python train.py input/trainData.csv output/model.pkl
```

The train command reads training data from a CSV file into a Pandas dataframe. It learns a linear regression from `rainfall` and `mean_temperature` (X) to `disease_cases` (Y). The trained model is stored to file using the joblib library:

```python
def train(train_data_path, model_path):
    df = pd.read_csv(train_data_path)
    features = df[["rainfall", "mean_temperature"]].fillna(0)
    target = df["disease_cases"].fillna(0)

    model = LinearRegression()
    model.fit(features, target)
    joblib.dump(model, model_path)
```

### Generating forecasts

Run the predict command to forecast disease cases based on future climate data and the previously trained model:

```bash
uv run python predict.py output/model.pkl input/trainData.csv input/futureClimateData.csv output/predictions.csv
```

The predict command loads the trained model and applies it to future climate data, outputting disease forecasts as a CSV file:

```python
def predict(model_path, historic_data_path, future_data_path, out_file_path):
    model = joblib.load(model_path)
    future_df = pd.read_csv(future_data_path)
    features = future_df[["rainfall", "mean_temperature"]].fillna(0)

    predictions = model.predict(features)
    output_df = future_df[["time_period", "location"]].copy()
    output_df["sample_0"] = predictions
    output_df.to_csv(out_file_path, index=False)
```

## Making model alterations

Here are some modifications you can try:

### Change the model type

Replace `LinearRegression` with a different sklearn model:

```python
# Original
from sklearn.linear_model import LinearRegression
reg = LinearRegression()

# Try Ridge regression instead
from sklearn.linear_model import Ridge
reg = Ridge(alpha=1.0)
```

### Add or remove features

Modify which columns are used as input features:

```python
# Original uses rainfall and mean_temperature
features = df[["rainfall", "mean_temperature"]]

# Try using only rainfall
features = df[["rainfall"]]
```

### Add data preprocessing

Add preprocessing steps like scaling:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
```

### Test your changes

After making changes, run the isolated test to verify everything works:

```bash
uv run python isolated_run.py
```

Check that:
- The script runs without errors
- Output files are generated in the `output/` directory

### Commit and push your changes

Once your modifications work, save them to your fork:

```bash
git add .
git commit -m "Modified model: [describe your change]"
git push origin main
```

## Running the minimalist model as part of CHAP

Running your model through CHAP gives you several benefits:

- You can easily run your model against standard evaluation datasets
- You can share your model with others in a standard way
- You can make your model accessible through the DHIS2 Modeling app

To run the minimalist model in CHAP, we define the model interface in a YAML specification file called `MLproject`. This file specifies that the model uses uv for environment management (`uv_env: pyproject.toml`) and defines the train and predict entry points:

```yaml
name: minimalist_example_uv

uv_env: pyproject.toml

entry_points:
  train:
    parameters:
      train_data: str
      model: str
    command: "python train.py {train_data} {model}"
  predict:
    parameters:
      historic_data: str
      future_data: str
      model: str
      out_file: str
    command: "python predict.py {model} {historic_data} {future_data} {out_file}"
```

After you have installed chap-core ([installation instructions](https://dhis2-chap.github.io/chap-core/chap-cli/chap-core-cli-setup.html)), you can run this minimalist model through CHAP as follows:

```bash
chap evaluate --model-name /path/to/minimalist_example_uv --dataset-name ISIMIP_dengue_harmonized --dataset-country brazil --report-filename report.pdf
```

**Parameters:**

- `--model-name`: Path to your local model directory (where MLproject is located)
- `--dataset-name`: The evaluation dataset to use
- `--dataset-country`: Country filter for the dataset
- `--report-filename`: Output PDF report

Or if you have a local CSV dataset:

```bash
chap evaluate --model-name /path/to/minimalist_example_uv --dataset-csv your_data.csv --report-filename report.pdf
```

## Creating your own model

You can use this example as a starting point for your own model. The key files are:

- `MLproject` - Defines how CHAP interacts with your model
- `pyproject.toml` - Lists your Python dependencies
- `train.py` - Contains the training logic
- `predict.py` - Contains the prediction logic
- `isolated_run.py` - Script to test the model without CHAP
- `input/` - Sample training and future climate data

To create a new model from scratch, you can use the `chap init` command:

```bash
chap init my_new_model
cd my_new_model
python isolated_run.py  # Test the model
```
