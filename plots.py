# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

def cross_validation_visualization(lambds, score_tr, score_te):
    """visualize the curves of score on training and score on tests."""
    plt.semilogx(lambds, score_tr, marker=".", color='b', label='train score');
    plt.semilogx(lambds, score_te, marker=".", color='r', label='test score');
    plt.xlabel("lambda")
    plt.ylabel("score")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation_test")
    

def degree_performance_visualization(degrees, scores):
    """Visualize the curve of score vs added degree."""
    plt.plot(scores, marker=".", color='b');
    plt.xticks(range(len(scores)))
    ax = plt.gca()
    ax.set_xticklabels(degrees)
    plt.xlabel("Degrees")
    plt.ylabel("Score")
    plt.title("Performance over degrees")
    plt.grid(True)
    plt.savefig("degree_performances_test")