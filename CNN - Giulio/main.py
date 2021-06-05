import tensorflow as tf
import sys
from data_loader import load_dataset, augmentation
from model import ResNet, EfficientNetB3
from solver import ImageClassifier
import wandb


training_data, training_labels, test_data, test_labels = load_dataset()
print('Data loaded')

training_aug_data, training_aug_labels = augmentation(training_data, training_labels)
print('Data augmented')

model = EfficientNetB3()
ic = ImageClassifier(batch_size=10, epochs=100)

wandb.init(project='ML challenge', group='training', name='challenge_2')
wandb.config.epochs = 100
wandb.config.batch_size = 10

print('Starting training')
ic.train(training_aug_data, training_aug_labels, test_data, test_labels, model)