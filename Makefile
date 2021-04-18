CURRENT_DIR = $(shell pwd)

.PHONY: tensorboard
tensorboard:
	tensorboard --logdir=experiments

.PHONY: mlflow 
mlflow:
	mlflow ui

.PHONY: test
test:
	pytest

