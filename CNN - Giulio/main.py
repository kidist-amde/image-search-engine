import tensorflow as tf
import sys
from data_loader import load_dataset, augmentation
from model import ResNet
from solver import ImageClassifier


training_data, training_labels, test_data, test_labels = load_dataset()

training_aug_data, training_aug_labels = augmentation(training_data, training_labels)

model = ResNet()
ic = ImageClassifier(batch_size=20, epochs=100)

ic.train(training_aug_data, training_aug_labels, test_data, test_labels, model)