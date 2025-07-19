TM1 Knowledge Navigator

This project is a practical example of fine-tuning large language models using LoRA (Low-Rank Adaptation) techniques. It includes scripts and data for generating synthetic samples, preprocessing, improving data quality, and training a model with LoRA.

Features:
- Tools for generating synthetic training data
- Scripts for data preprocessing and quality checks
- Training workflow for LoRA-based model fine-tuning
- Configuration files for reproducibility

Getting Started:
1. Install dependencies listed in pyproject.toml
2. Generate synthetic data: python syntheticdatageneration.py
3. Preprocess data: python preprocessing.py
4. (Optional) Run dataquality.py to improve data
5. Train the model: python train.py

Project Structure:
- data/: Contains sample and processed data
- Main scripts: before.py, generated_prompt.py, preprocessing.py, syntheticdatageneration.py, train.py
- pyproject.toml: Project dependencies and configuration

