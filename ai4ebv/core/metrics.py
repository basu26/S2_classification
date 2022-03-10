"""Accuracy assessment."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# externals
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def area_adjusted_classification_report(y_true, y_pred, y, labels=None,
                                        target_names=None, res=30):
    """Area adjusted classification accuracy after `Olofsson et al. (2014)`_.


    Parameters
    ----------
    y_true : :py:class:`numpy.ndarray`, (nsamples)
        Reference data.
    y_pred : :py:class:`numpy.ndarray`, (nsamples,)
        Model classifications.
    y : :py:class:`numpy.ndarray`
        Classification map. Used to calculate area adjustment.
    labels : `list` or :py:class:`numpy.ndarray`, optional
        Array of class labels. If not specified, the class labels are inferred
        from ``y_true``.
    target_names : `list`, optional
        Array of class names. If specified, the output
        :py:class:`pandas.DataFrame` is annotated with the class names.
    res : `int`, optional
        Spatial resolution of the classified image. The default is 30.

    Returns
    -------
    df : :py:class:`pandas.DataFrame`
        Class-wise area adjusted classification metrics.

    .. _Olofsson et al. (2014):
        https://www.sciencedirect.com/science/article/abs/pii/S0034425714000704

    """
    # calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize=None, labels=labels)

    # number of classes
    if labels is None:
        labels = np.unique(y_true)

    # calculate area of each class
    maparea = np.asarray([np.sum(y == c) * (res ** 2) for c in labels])

    # total map area
    A = np.sum(maparea)

    # proportion of area mapped by each class
    W = maparea / A

    # number of reference points per class
    n_i = cm.sum(axis=1)

    # area adjusted confusion matrix
    p = W * (cm / n_i)

    # overall accuracy
    oa = np.sum(np.diag(p))

    # producer's accuracy (recall) and omission error
    pa = np.diag(p) / p.sum(axis=1)
    oe = 1 - pa

    # user's accuracy (precision) and commission error
    ua = np.diag(p) / p.sum(axis=0)
    ce = 1 - ua

    # F1-score
    f1 = 2 * (pa * ua) / (pa + ua)

    # classification report
    report = np.stack([ua, pa, f1, ce, oe, W, n_i], axis=1)

    # construct dataframe
    df = pd.DataFrame(report, columns=('precision', 'recall', 'f1-score',
                                       'commission', 'omission', 'area',
                                       'support'),
                      index=target_names)

    # compute macro average and weighted average
    m_avg = df.sum(axis=0) / len(labels)
    w_avg = df.multiply(df['area'], axis='index').sum(axis=0)

    # append to dataframe
    df.loc['weighted avg'] = w_avg
    df.loc['macro avg'] = m_avg

    # add row for overall accuracy
    df.loc['accuracy'] = oa

    # area and support considered for weighted/macro average
    df.area[-3:] = 1
    df.support[-3:] = df.support[:-3].sum()

    return df
