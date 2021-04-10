CURRENT_DIR = $(shell pwd)

.PHONY: tensorboard
tensorboard:
	tensorboard --logdir=experiments

.PHONY: test
test:
	pytest

