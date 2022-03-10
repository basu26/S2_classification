"""Scikit-learn model and training configuration."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# externals
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

# define the classifier
# clf = SVC(probability=True)
clf = RandomForestClassifier(n_jobs=-1)

# define the hyperparameter grid to search
rdf_grid = {'clf__n_estimators': [50, 500, 1000],
            'clf__min_samples_split': np.arange(2, 5)}
svm_grid = {'clf__C': [0.1, 1, 10],
            'clf__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            }

# select matching grid
grid = rdf_grid if isinstance(clf, RandomForestClassifier) else svm_grid

# define pipeline: scaling followed by classification
pipe = Pipeline([('scale', MinMaxScaler()), ('clf', clf)])

# define grid search
clf = GridSearchCV(pipe, param_grid=grid, n_jobs=-1, refit=True, verbose=10,
                   scoring='f1_macro')

# inference block size
TILE_SIZE = (1024, 1024)

# overwrite output layers
SKLEARN_OVERWRITE = False
