# Schneider Electric Europe Data Challenge

## Overview

With the increasing digitalization and the growing reliance on data servers, Schneider Electric presents an innovative challenge to predict the European country with the highest surplus of green energy in the next hour. This prediction is crucial for optimizing computing tasks, utilizing green energy effectively, and reducing CO2 emissions.

## Objective

Create a model to predict the European country with the highest surplus of green energy in the next hour. Consider energy generation from renewable sources and energy consumption. The solution should align with Schneider Electric's ethos and present an unprecedented approach.

## Dataset

Utilize time-series data from the ENTSO-E Transparency portal API, including electricity consumption, wind energy generation, solar energy generation, and other green energy generation. Homogenize the data to 1-hour intervals, and create 'train.csv' and 'test.csv' datasets from 01-01-2022 to 01-01-2023.

API Token: `1d9cd4bd-f8aa-476c-8cc1-3442dc91506d` (or alternative tokens provided)

## Repository Structure

```plaintext
|__README.md
|__requirements.txt
|
|__data
|  |__train.csv
|  |__test.csv
|
|__src
|  |__data_ingestion.py
|  |__data_processing.py
|  |__model_training.py
|  |__model_prediction.py
|  |__utils.py
|
|__models
|  |__model.pkl
|
|__scripts
|  |__run_pipeline.sh
|
|__predictions
   |__predictions.json
```

## Installation and Usage

1. **Clone the Repository:**

```bash
   git clone https://github.com/s-matke/eco-forecast.git
   cd eco-forecast
```
2. **Install Dependencies:**
```bash
   pip install -r requirements.txt
```
3. **Run Pipeline:**
```bash
  ./run_pipeline.sh <start_date> <end_date> <raw_data_file> <processed_data_file> <model_file> <test_data_file><predictions_file>
```
For example:
```bash
  ./run_pipeline.sh 2022-01-01 2023-01-01 data/raw_data.csv data/processed_data.csv models/model.pkl data/test_data.csv predictions/predictions.json
```

## Data Processing

- Missing values are imputed as the mean between preceding and following values.
- Data with resolution finer than 1 hour is resampled to an hourly level.
- Non-green energy sources are discarded.
- The resulting CSV includes columns per country representing generated green energy per energy type and load.
- Calculated surplus as the difference between generated green energy and load in order to create label feature
- Downsampled labels that are dominating, however, it's set as an optional parameter

## Model

Used Multi-layer Perceptron (MLP) classifier as our model from scikit-learn module to predict the country with the highest surplus green energy. The model is saved inside models directory while it's predictions on test dataset have been saved in predictions directory.
