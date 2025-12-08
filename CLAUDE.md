# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GoLearn is a machine learning library for Go that implements the scikit-learn Fit/Predict interface pattern. The module path is `github.com/sjwhitworth/golearn`.

## Build and Test Commands

```bash
# Run all tests with coverage
./coverage.sh

# Run tests for a specific package
go test ./base/...
go test ./knn/...

# Run a single test
go test -run TestName ./package/...

# Get dependencies
go get -v ./...
```

## System Dependencies

The `linear_models` package requires liblinear to be installed:
```bash
# Ubuntu/Debian
sudo apt-get install libatlas-base-dev
# Then build and install liblinear-1.94 from source
```

Set `GODEBUG=cgocheck=0` when running tests involving cgo.

## Architecture

### Core Abstractions (base/)

**DataGrid interfaces** - The fundamental data abstraction (similar to DataFrames):
- `DataGrid` - Basic row/column addressable data with Attributes
- `FixedDataGrid` - DataGrid with known size, used for model input/output
- `UpdatableDataGrid` - Mutable DataGrid for building datasets

**Classifier interface** - All classifiers implement:
```go
type Classifier interface {
    Fit(FixedDataGrid) error
    Predict(FixedDataGrid) (FixedDataGrid, error)
    Save(string) error
    Load(string) error
    // ...
}
```

**Attributes** - Define column types: `FloatAttribute`, `CategoricalAttribute`, `BinaryAttribute`

### ML Algorithm Packages

- `knn/` - K-nearest neighbors (supports euclidean, manhattan, cosine distances; linear and kdtree search)
- `trees/` - Decision trees (ID3, CART classifier/regressor), Isolation Forest for anomaly detection
- `ensemble/` - Random Forest, Multi-class SVC
- `linear_models/` - Linear regression, logistic regression, liblinear SVM wrapper (requires cgo)
- `neural/` - Feed-forward neural networks
- `perceptron/` - Averaged perceptron
- `clustering/` - DBSCAN, Expectation-Maximization

### Support Packages

- `evaluation/` - ConfusionMatrix, precision/recall/F1, cross-validation (`GenerateCrossFoldValidationConfusionMatrices`)
- `filters/` - Data preprocessing: binning, chi-merge discretization, binary conversion
- `pca/` - Principal Component Analysis
- `kdtree/` - KD-tree implementation for efficient nearest neighbor search

### Data Loading

```go
// Load CSV with headers
rawData, err := base.ParseCSVToInstances("path/to/data.csv", true)

// Train/test split
trainData, testData := base.InstancesTrainTestSplit(rawData, 0.50)
```

### Typical Usage Pattern

```go
cls := knn.NewKnnClassifier("euclidean", "linear", 2)
cls.Fit(trainData)
predictions, _ := cls.Predict(testData)
confusionMat, _ := evaluation.GetConfusionMatrix(testData, predictions)
fmt.Println(evaluation.GetSummary(confusionMat))
```
