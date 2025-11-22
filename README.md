# Materials Property Prediction: Transformer vs Baseline Architectures

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Author:** Ibtisam Ahmed Khan  
**Affiliation:** Tsinghua University  
**Date:** November 2025

## 🎯 Project Overview

This project investigates the application of Transformer architectures, traditionally used in natural language processing, to materials science for predicting electronic band gaps from chemical composition. The work demonstrates a novel physics-informed training approach and provides comparative analysis against baseline methods.

### Key Contributions

1. **Novel Tokenization Strategy**: Developed a method to represent chemical formulas as token sequences for Transformer processing
2. **Physics-Informed Loss Function**: Implemented custom loss incorporating physical constraints (band gap ≥ 0)
3. **Comprehensive Benchmarking**: Systematic comparison of Transformer vs baseline MLP architectures
4. **Critical Analysis**: Identified conditions where different architectures excel

## 📊 Results Summary

| Model | MAE (eV) | R² Score | Parameters |
|-------|----------|----------|------------|
| **Baseline MLP** | **0.8665** | **0.4817** | ~50K |
| **Transformer** | 0.9373 | 0.4118 | ~180K |

**Dataset:** 10,000 materials from Materials Project database  
**Target Property:** Electronic band gap (0-10 eV range)

## 🔬 Methodology

### 1. Data Collection
- Source: [Materials Project](https://materialsproject.org/) API
- Materials: 10,000 inorganic compounds with known band gaps
- Features: Chemical composition (element types and stoichiometry)
- Target: Electronic band gap (eV)

### 2. Tokenization Approach

Chemical formulas are converted to sequences of element tokens:

```
Example: Fe₂O₃ → [<START>, Fe, Fe, O, O, O, <END>]
```

- **Vocabulary Size:** 86 unique element tokens
- **Maximum Sequence Length:** 50 tokens
- **Special Tokens:** `<PAD>`, `<START>`, `<END>`

### 3. Model Architectures

#### Transformer Model
- **Architecture:** Multi-head self-attention encoder
- **Layers:** 4 transformer blocks
- **Attention Heads:** 8
- **Embedding Dimension:** 128
- **Total Parameters:** ~180,000
- **Key Features:**
  - Positional encoding for sequence awareness
  - Multi-head attention for capturing element interactions
  - Global average pooling over sequence
  - Regression head for property prediction

#### Baseline MLP
- **Architecture:** Simple feedforward network
- **Layers:** 4 fully connected layers [64→128→64→32→1]
- **Pooling:** Simple mean pooling over embeddings
- **Total Parameters:** ~50,000

### 4. Physics-Informed Training

Custom loss function combining prediction accuracy with physical constraints:

```
L_total = MSE(y_pred, y_true) + α × ReLU(-y_pred)
```

Where:
- **MSE term:** Standard mean squared error for accuracy
- **Physics penalty:** Penalizes negative band gap predictions (α=0.1)
- **Result:** All predictions remain physically valid (≥0 eV)

### 5. Training Details
- **Optimizer:** Adam (lr=0.001)
- **Scheduler:** ReduceLROnPlateau
- **Batch Size:** 128
- **Epochs:** 20
- **Data Split:** 70% train, 15% validation, 15% test
- **Hardware:** Google Colab (Tesla T4 GPU)

## 📈 Analysis and Insights

### Performance Comparison

The baseline MLP outperformed the Transformer by 8.2% in MAE. This result provides important insights:

#### Why Baseline Performed Better

1. **Dataset Size Limitation**
   - 10,000 samples insufficient for Transformer's capacity
   - Transformers typically require 100K+ samples to excel
   - Baseline better suited for small data regime

2. **Task Complexity**
   - Band gap prediction from composition is relatively simple
   - No long-range dependencies requiring self-attention
   - Element interactions mostly local (nearest neighbors)

3. **Model Capacity vs Data**
   - Transformer: 180K parameters, 10K samples → potential overfitting
   - Baseline: 50K parameters, 10K samples → better regularization

4. **Feature Representation**
   - Simple element sequences may not capture crystal structure
   - Baseline mean pooling sufficient for composition-level features

### When Would Transformers Excel?

Based on this analysis, Transformers would likely outperform on:

1. **Larger Datasets:** 50K+ materials samples
2. **Complex Structures:** 2D/3D spatial arrangements (e.g., metamaterials geometry)
3. **Multi-state Systems:** Reconfigurable materials with state-dependent properties
4. **Long-range Dependencies:** Properties requiring understanding of distant element interactions
5. **Inverse Design:** Generating structures from target properties (generative task)

### Value of Physics-Informed Approach

Despite lower accuracy, the physics-informed Transformer achieved:
- **100% physical validity:** No negative band gap predictions
- **Constraint satisfaction:** Critical for inverse design applications
- **Interpretability:** Physics penalties track learning progress

## 🎓 Relevance to Doctoral Research

This project serves as a proof-of-concept for my proposed PhD research on **"Generative Inverse Design of Reconfigurable Metamaterials using Physics-Informed Large Language Models"** at Tsinghua University.

### Direct Applications to PhD Work

1. **Tokenization Strategy:** Foundation for representing 2D/3D nanostructures
2. **Physics-Informed Loss:** Embedding Maxwell's Equations for photonic design
3. **Architecture Understanding:** Knowing when Transformers add value vs simpler models
4. **Benchmarking Methodology:** Framework for comparing AI approaches

### Lessons for Metamaterials Design

- Transformers likely superior for spatial metamaterial patterns (vs simple composition)
- Larger synthetic datasets (100K+) needed from FDTD simulations
- Physics constraints essential for manufacturability
- Architecture choice depends on problem complexity

## 🔄 Future Improvements

### Short-term Enhancements
1. **Increase Dataset Size:** Collect 50K+ materials for proper Transformer training
2. **Add Crystal Structure:** Include lattice parameters, space group symmetry
3. **Hyperparameter Tuning:** Grid search over learning rate, model depth, attention heads
4. **Attention Visualization:** Analyze which element interactions Transformer captures

### Long-term Extensions
1. **Multi-Property Prediction:** Simultaneously predict band gap, formation energy, stability
2. **Generative Design:** Inverse model predicting composition from target properties
3. **Graph Neural Networks:** Represent materials as atomic graphs
4. **Transfer to Photonics:** Apply methodology to optical metamaterials design

## 🛠️ Technical Implementation

### Requirements
```
python>=3.8
torch>=2.0.0
mp-api>=0.41.0
pymatgen>=2023.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

### Installation
```bash
git clone https://github.com/yourusername/materials-transformer-bandgap-prediction.git
cd materials-transformer-bandgap-prediction
pip install -r requirements.txt
```

### Usage
```python
# Get free API key from materialsproject.org
API_KEY = "your_api_key_here"

# Run the complete pipeline
python train.py --api_key $API_KEY --num_materials 10000 --epochs 20
```

Or use the provided Jupyter notebook for interactive exploration.

## 📁 Repository Structure
```
materials-transformer-bandgap-prediction/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── notebook.ipynb                     # Complete implementation
├── results/
│   ├── bandgap_distribution.png      # Dataset visualization
│   └── results_comprehensive.png     # Training curves and predictions
└── models/
    ├── best_transformer.pt           # Trained model weights
    └── best_baseline.pt              # Baseline weights
```

## 📚 References

**Materials Project:**
- Jain, A., et al. (2013). Commentary: The Materials Project: A materials genome approach to accelerating materials innovation. *APL Materials*, 1(1), 011002.

**Transformer Architecture:**
- Vaswani, A., et al. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30.

**Physics-Informed Neural Networks:**
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems. *Journal of Computational Physics*, 378, 686-707.

**Materials Informatics:**
- Xie, T., & Grossman, J. C. (2018). Crystal graph convolutional neural networks for accurate and interpretable prediction of material properties. *Physical Review Letters*, 120(14), 145301.

## 🤝 Contributing

This project is part of my PhD application portfolio. Feedback and suggestions are welcome:
- **Issues:** Report bugs or suggest improvements
- **Pull Requests:** Contributions to extend functionality
- **Discussions:** Share ideas for applications to other materials domains

## 📧 Contact

**Ibtisam Ahmed Khan**  
- Email: khanibtisam38@gmail.com
- LinkedIn: https://www.linkedin.com/in/ibtisam-khan/
- GitHub: ibtisamkhan96

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Materials Project team for providing open-access database and API
- PyTorch team for deep learning framework
- AtomCamp for data science training
- Tsinghua University for research opportunities

---

**Note:** This project demonstrates technical proficiency, critical thinking, and scientific rigor - key qualities for doctoral research. The results highlight the importance of matching model complexity to problem requirements, a valuable lesson for AI-driven materials discovery.
