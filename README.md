# AWS Credit Card Fraud Detection

Projet AWS MLOps pour la détection de fraude bancaire.

## Objectif

Construire un pipeline complet :

1. Stockage des données brutes dans Amazon S3
2. ETL avec SageMaker Processing
3. Analyse de biais avec SageMaker Clarify
4. Entraînement avec SageMaker Training
5. Évaluation du modèle
6. Déploiement sur SageMaker Serverless Endpoint
7. Monitoring avec SageMaker Model Monitor
8. CI/CD avec GitHub, CodePipeline et CodeBuild

## Architecture

GitHub contient le code du projet.

Amazon S3 contient les datasets, les données transformées, les modèles et les rapports.

SageMaker exécute l’ETL, l’entraînement, l’évaluation et le déploiement.
