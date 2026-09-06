# Customer Intelligence Platform

Week 3 Project -- Unsupervised Learning and ML Pipelines

> Production-grade customer segmentation and churn prediction system combining silhouette-validated K-Means/DBSCAN clustering with a leak-free sklearn Pipeline, MLflow-tracked experiments, and Model Registry-based promotion — served via FastAPI and CLI batch inference, containerized with Docker for reproducible deployment.

## Overview
Customer Intelligence Platform is an end-to-end unsupervised-plus-supervised learning system that turns raw customer transaction/behavioral data into two production artefacts: a segmentation model and a churn classifier, both served through a single, leak-free `sklearn.pipeline.Pipeline`.

The segmentation stage evaluates K-Means, DBSCAN, and agglomerative clustering against silhouette score and cluster stability under bootstrap resampling, rather than accepting the first converged solution — DBSCAN's density-based clusters are used specifically to surface non-convex customer groups that K-Means' centroid geometry cannot represent. PCA and UMAP are applied purely for diagnostic visualization of cluster separation in 2D, never as a modeling input, keeping the feature space interpretable for the churn model downstream.

Feature engineering and encoding are isolated inside a `ColumnTransformer` (numeric imputation + scaling, categorical one-hot encoding) composed into a single `Pipeline` object with the final estimator, so the exact same fitted transformer state is guaranteed at train and inference time — eliminating the train/serve skew that ad hoc pandas preprocessing scripts are prone to. Every experiment — cluster count, algorithm, hyperparameter grid, and churn-model variant — is logged to MLflow Tracking (parameters, metrics, confusion matrices, and the serialized pipeline artefact), with the best-performing, gate-passing churn model promoted through the MLflow Model Registry's stage lifecycle (`Staging` → `Production`) rather than tracked by convention or file naming.

Inference is exposed two ways: a FastAPI service for real-time single-customer scoring and a CLI entry point for scheduled batch prediction jobs, both loading the registry's `Production`-stage model rather than a hardcoded path or version, so promoting a retrained model requires no code change. The whole stack — training, tracking server, and serving — is containerized via Docker for reproducible deployment across environments.

## Key Concepts Applied
- K-Means, DBSCAN, Hierarchical clustering
- PCA, t-SNE, UMAP
- sklearn Pipeline, ColumnTransformer
- MLflow experiment tracking

## Tech Stack
- Python, scikit-learn, MLflow
- Docker

## Quick Start

```bash
make setup      # create venv and install deps
make train      # run training pipeline
make track      # launch MLflow UI
make predict    # run CLI batch predictions
make test       # run tests
```

## Status
Not started
