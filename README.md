# Traffic Flow Forecasting on Spatio-Temporal Data

[![PDF](https://img.shields.io/badge/PDF-View%20Paper-red?style=flat-square&logo=adobe-acrobat-reader)](./MagNet_B%20(1).pdf)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-orange?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

## Overview

This repository contains the implementation and research findings for **Traffic Flow Forecasting on Spatio-Temporal Data** - a comprehensive study exploring graph-based deep learning approaches for modeling traffic dynamics across space and time using real-world datasets from multiple cities.

**Authors:** Daniel Ochana, Michal Maya  
**Supervisor:** Ya-Wei Eileen Lin  
**Date:** April 2025

## 🎯 Key Contributions

- **Novel MagNet Adaptation**: First application of MagNet Graph Neural Networks to weighted directed graphs for traffic forecasting
- **Dual Representation Strategy**: Comparison of symmetric vs. non-symmetric adjacency representations for spatial-temporal data
- **Multi-Dataset Evaluation**: Comprehensive testing across NYC Taxi, PeMSD8, and METR-LA datasets
- **Graph Construction Pipeline**: Innovative approach to representing traffic data as weighted directed graphs with time-slotted edges

## 📊 Results

Our best-performing model achieved:
- **86.0% accuracy** for "into traffic" classification
- **85.9% accuracy** for "from traffic" classification
- **Symmetric MagNet** consistently outperformed non-symmetric variants

## 🏗️ Architecture

### Graph Representation
- **Nodes**: Spatial locations (longitude, latitude coordinates)
- **Edges**: Road connections between locations
- **Weights**: Average driving speed in specific time slots
- **Temporal Dimension**: 18 different 3-hour time slots across 3 days

### Models Investigated
1. **MagNet Graph Neural Networks** - Primary focus with novel adaptations
2. **DyRep** - Dynamic graph representation learning
3. **ST-SSL** - Spatio-Temporal Self-Supervised Learning

## 📁 Repository Structure

```
├── MagNet_B (1).pdf           # Research paper
├── src/
│   ├── models/
│   │   ├── magnet.py          # MagNet implementation
│   │   ├── dyrep.py           # DyRep model
│   │   └── st_ssl.py          # Self-supervised learning
│   ├── data/
│   │   ├── preprocessing.py   # Data preprocessing pipeline
│   │   └── graph_construction.py
│   ├── utils/
│   │   ├── evaluation.py      # Model evaluation metrics
│   │   └── visualization.py   # Result visualization
│   └── experiments/
│       ├── node_classification.py
│       └── link_prediction.py
├── data/                      # Dataset directory
├── results/                   # Experimental results
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Required Libraries
- PyTorch >= 1.9.0
- PyTorch Geometric
- NumPy
- Pandas
- NetworkX
- Matplotlib
- Scikit-learn

### Dataset Preparation
1. Download the NYC Taxi dataset (January 2016)
2. Run preprocessing pipeline:
```bash
python src/data/preprocessing.py --dataset nyc_taxi
```

### Training
```bash
# Train MagNet (Symmetric)
python src/experiments/node_classification.py --model magnet --variant symmetric

# Train MagNet (Non-symmetric)
python src/experiments/node_classification.py --model magnet --variant non_symmetric
```

## 📈 Methodology

### Magnetic Laplacian Approach
We adapt the Magnetic Laplacian matrix for weighted directed graphs:

```
L_q = D_s - H_q = D_s - A_s · exp(iΘ_q)
```

Where:
- `A_s`: Symmetrized adjacency matrix
- `Θ_q`: Phase matrix encoding directional information
- `D_s`: Corresponding degree matrix

### Two Key Adaptations

1. **Symmetric Version**: Directional information only in phase matrix
2. **Non-symmetric Version**: Both phase and adjacency matrices retain weighted directional information

## 🔬 Experimental Setup

### Datasets
- **NYC Taxi Dataset**: January 2016 trip data with pickup/dropoff coordinates
- **PeMSD8**: Highway sensor data from California
- **METR-LA**: Urban traffic monitoring from Los Angeles

### Evaluation Tasks
- **Node Classification**: Predicting traffic flow direction (into/from nodes)
- **Link Prediction**: Estimating edge weights (average speeds) over time

### Performance Metrics
- Classification accuracy
- Cross-entropy loss
- Temporal consistency measures

## 📊 Results Summary

| Method | Labels | Train Accuracy | Test Accuracy |
|--------|--------|----------------|---------------|
| Non-Symmetric | into | 85.7% | 84.8% |
| Non-Symmetric | from | - | 84.1% |
| **Symmetric** | **into** | **86.9%** | **86.0%** |
| **Symmetric** | **from** | **84.5%** | **85.9%** |

## 🔮 Future Work

- **Temporal Edge Integration**: Incorporate cross-time-slot connections
- **Advanced Dynamic Models**: Further exploration of DyRep and ST-SSL adaptations
- **Multi-city Generalization**: Extend to other urban transportation networks
- **Real-time Implementation**: Deploy for live traffic forecasting systems

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@article{ochana2025traffic,
  title={Traffic Flow Forecasting on Spatio-Temporal Data},
  author={Ochana, Daniel and Maya, Michal},
  supervisor={Lin, Ya-Wei Eileen},
  year={2025},
  month={April}
}
```

## 📚 References

1. Zhang, X., He, Y., Brugnone, N., Perlmutter, M., & Hirn, M. (2021). MagNet: A neural network for directed graphs. *Advances in Neural Information Processing Systems*, 34, 27003-27015.

2. Trivedi, R., Farajtabar, M., Biswal, P., & Zha, H. (2021). DyRep: Learning representations over dynamic graphs. *arXiv preprint*.

3. Anonymous. (2021). Spatio-temporal self-supervised learning for traffic forecasting. *OpenReview*.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit pull requests, report issues, and contribute to the project.

## 📧 Contact

- **Daniel Ochana** - [GitHub](https://github.com/danielochana) | [Email](mailto:danielochana@gmail.com)
- **Michal Maya** - [GitHub](https://github.com/michalmaya) | [Email](mailto:michal.maya75@gmail.com)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

⭐ **Star this repository** if you find it helpful for your traffic forecasting research!
