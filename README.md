# NAICS-Aware Graph Neural Networks for Large-Scale POI Co-visitation Prediction

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Conference](https://img.shields.io/badge/conference-KDD%202025-red)](https://kdd.org/)

[Paper](https://arxiv.org/abs/2507.19697) | [Dataset](https://huggingface.co/datasets/alrubyli/Covisit)

## 📋 Abstract

Understanding where people go after visiting one business is crucial for urban planning, retail analytics, and location-based services. However, predicting these co-visitation patterns across millions of venues remains challenging due to extreme data sparsity and the complex interplay between spatial proximity and business relationships. We introduce **NAICS-aware GraphSAGE**, a novel graph neural network that integrates business taxonomy knowledge through learnable embeddings to predict population-scale co-visitation patterns.

Our key insight is that business semantics—captured through detailed industry codes—provide crucial signals that pure spatial models cannot explain. The approach scales to massive datasets (4.2 billion potential venue pairs) through efficient state-wise decomposition while combining spatial, temporal, and socioeconomic features in an end-to-end framework.

## 🔥 Key Contributions

1. **Methodological Innovation**: First end-to-end GNN framework that jointly embeds NAICS codes, temporal signals, and spatial relations for population-level co-visitation prediction through edge regression

2. **Strong Performance**: Achieves test R² of 0.625 (157% improvement over best baseline) with significant gains in ranking quality (32% improvement in NDCG@10)

3. **Large-Scale Dataset**: We release **POI-Graph**, comprising:
   - 94.9 million co-visitation records
   - 45.3 million graph edges
   - 92,486 brands across 48 US states
   - 276 NAICS business categories
   - 38 socioeconomic indicators

## 📊 Dataset: POI-Graph

### Overview
POI-Graph is the first large-scale dataset specifically designed for co-visitation research, enabling reproducible advances in mobility modeling and urban analytics.

### Statistics
| Component | Count |
|-----------|-------|
| Co-visitation Records | 94.9M |
| Graph Edges | 45.3M |
| Unique Brands | 92,486 |
| US States Covered | 48 |
| Business Categories | 276 |
| Socioeconomic Features | 38 |
| Time Period | Jan 2018 - Mar 2020 |

### Data Schema
```
poi_graph/
├── graphs/           # State-wise graph structures
│   ├── TX/          # Texas graph data
│   ├── CA/          # California graph data
│   └── ...
├── features/         # Node and edge features
│   ├── naics/       # NAICS business categories
│   ├── temporal/    # Time-series features
│   └── socioeconomic/ # Census block group data
├── co_visits/       # Raw co-visitation counts
└── metadata/        # Brand information and mappings
```

## 🛠️ Installation

### Requirements
- Python 3.8+
- PyTorch 1.12+
- PyTorch Geometric 2.3+
- CUDA 11.3+ (for GPU support)

### Setup

```bash
# Clone the repository
git clone https://github.com/yazeedalrubyli/poi-covisitation-prediction
cd poi-covisitation-prediction

# Create conda environment
conda create -n poi-covisit python=3.8
conda activate poi-covisit

# Install dependencies
pip install -r requirements.txt

# Download the POI-Graph dataset (12.3 GB)
python scripts/download_data.py --dataset poi-graph
```

## 🚀 Quick Start

### Training a Model

```python
from naics_graphsage import NAICSGraphSAGE
from data_loader import POIGraphDataset

# Load dataset
dataset = POIGraphDataset(
    root='./data/poi_graph',
    state='TX',  # Texas as example
    lookback_months=12
)

# Initialize model
model = NAICSGraphSAGE(
    input_dim=dataset.num_features,
    hidden_dim=128,
    naics_vocab_size=276,
    naics_embed_dim=64
)

# Train
trainer = Trainer(model, dataset)
trainer.fit(epochs=100, lr=0.001)
```

### Making Predictions

```python
# Load trained model
model = NAICSGraphSAGE.load_pretrained('models/best_model.pt')

# Predict co-visitation between two brands
prediction = model.predict_covisit(
    brand_a='Starbucks',
    brand_b='Chipotle',
    state='CA',
    month='2020-01'
)
print(f"Predicted co-visits: {prediction:.0f}")
```

## 📈 Results

### Performance Comparison

| Method | Test R² | RMSE | NDCG@10 | MAE |
|--------|---------|------|---------|-----|
| Gravity Model | -0.04 | 35.3 | 0.25 | 6.7 |
| GeoMF++ | -0.05 | 35.5 | 0.23 | 7.6 |
| LightGBM | 0.04 | 34.0 | 0.34 | 8.5 |
| STHGCN | 0.243 | 30.2 | 0.52 | 5.5 |
| **NAICS-GraphSAGE (Ours)** | **0.625** | **28.5** | **0.687** | **5.2** |

### Key Findings

- **Business semantics matter**: NAICS embeddings contribute 23% to overall performance
- **Scalability**: Processes state-level graphs with 1.3M edges in under 2 hours
- **Interpretability**: Learned embeddings cluster semantically similar businesses

## 🔬 Reproducibility

### Training Scripts

```bash
# Reproduce main results
python scripts/train_all_states.py --config configs/main_experiment.yaml

# Run ablation studies
python scripts/ablation_study.py --ablation naics_embeddings

# Generate visualizations
python scripts/visualize_results.py --model models/best_model.pt
```

### Pre-trained Models

We provide pre-trained models for all 48 states:

```bash
# Download pre-trained models
python scripts/download_models.py --state all
```

### Experiment Logs

Complete training logs and experimental outputs are provided for full reproducibility:

- `output.log` - Comprehensive training logs including hyperparameter settings, validation metrics, and convergence details for all experiments reported in the paper

## 📖 Citation

If you find this work useful, please cite our paper:

<!-- ```bibtex
@inproceedings{alrubyli2024naics,
  title={NAICS-Aware Graph Neural Networks for Large-Scale POI Co-visitation Prediction: A Multi-Modal Dataset and Methodology},
  author={Alrubyli, Yazeed and Wafa, Abrar and Alomeir, Omar and Bahrami, Mohsen and Alrasheed, Hend and Hidvégi, Diána},
  booktitle={Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={xxx--xxx},
  year={2025},
  publisher={ACM}
}
``` -->

```bibtex
@misc{alrubyli2025naics,
  title={NAICS-Aware Graph Neural Networks for Large-Scale POI Co-visitation Prediction: A Multi-Modal Dataset and Methodology},
  author={Alrubyli, Yazeed and Alomeir, Omar and Wafa, Abrar and Hidvégi, Diána and Alrasheed, Hend and Bahrami, Mohsen},
  year={2025},
  eprint={2507.19697},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2507.19697}
}
```

### Areas for Extension
- [ ] Real-time prediction capabilities
- [ ] Additional business taxonomy systems (SIC, custom)
- [ ] Cross-city transfer learning
- [ ] Integration with traffic data

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Yazeed Alrubyli** - *Università di Bologna* - [yazeednaif.alrubyli2@unibo.it](mailto:yazeednaif.alrubyli2@unibo.it)
- **Omar Alomeir** - *Prince Sultan University*
- **Abrar Wafa** - *Prince Sultan University*
- **Diána Hidvégi** - *Intelmatix*
- **Hend Alrasheed** - *Massachusetts Institute of Technology*
- **Mohsen Bahrami** - *Massachusetts Institute of Technology*
---

For questions and feedback, please open an issue or contact the corresponding author. 
