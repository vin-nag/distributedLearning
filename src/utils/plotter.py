
import matplotlib.pyplot as plt
import numpy as np
import sys
import os


resultsNodes = {
    "One Node": ([95.41, 97.37, 98.12, 98.39, 98.64], 396.6382),
    "Three Nodes": ([92.15, 94.17, 95.16, 96.19, 96.57], 320.1437),
    "Seven Nodes": ([89.52, 91.1, 92.56, 94.19, 94.82], 297.7351),
    "Ten Nodes": ([85.37, 88.83, 90.79, 91.4, 92.65], 282.1651)
}


resultsEqual = {
    "average":([87.58, 89.83, 92.1, 92.6, 93.51], 297.4415),
    "weighted": ([89.42, 89.93, 92.0, 93.45, 94.02], 302.6039),
}

resultsRandom = {
    "average": ([85.37, 88.83, 90.79, 91.4, 92.65], 282.1651),
    "weighted": ([86.88, 89.58, 90.42, 92.15, 93.37], 285.5931)
}

resultsClass = {
    "average": ([9.8]*5, 293.7445),
    "weighted": ([9.8]*5, 296.3214)
}

resultsAverage = {
    "equal": ([87.58, 89.83, 92.1, 92.6, 93.65], 282.1651),
    "random": ([85.37, 88.37, 90.79, 91.4, 92.65], 297.4415),
    "class": ([9.8] * 5, 293.7445),
}

resultsWeighted = {
    "equal": ([89.42, 89.93, 92.0, 93.45, 94.02], 285.5931),
    "random": ([86.88, 89.58, 90.42, 92.15, 93.37], 302.6039),
    "class": ([9.8] * 5, 296.3214)
}

def printNodeAccuracies():
    plt.figure(figsize=(10,10))
    plt.xlabel("Training Epochs (#)", fontsize=18)
    plt.ylabel("Accuracy (%)", fontsize=18)
    x_axis = [(x+1) for x in range(5)]

    for result in resultsNodes:
        plt.plot(x_axis, resultsNodes[result][0], label=result)

    plt.xticks(x_axis)
    plt.ylim(75, 100)
    plt.grid(False)
    plt.legend(fontsize=20, loc=4)

    plt.savefig("./../results/nodeAccuracies.png")


def printOneAccuracy():
    plt.figure(figsize=(10,10))
    plt.xlabel("Training Epochs (#)", fontsize=18)
    plt.ylabel("Accuracy (%)", fontsize=18)
    x_axis = [(x+1) for x in range(5)]

    for result in resultsAverage:
        plt.plot(x_axis, resultsAverage[result][0], label=result)

    plt.xticks(x_axis)
    plt.ylim(0, 100)
    plt.grid(False)
    plt.legend(fontsize=20, loc=4)

    plt.savefig("./../results/averageAccuracies.png")

def printNodeTimes():
    plt.figure(figsize=(10,10))

    bars = ("One Node", "Three Nodes", "Seven Nodes", "Ten Nodes")
    y_pos = np.arange(len(bars))
    height = [resultsNodes[result][1] for result in resultsNodes.keys()]

    plt.bar(y_pos, height)
    plt.xlabel("Number of Worker Nodes (#)", fontsize=18)
    plt.ylabel("Training Time For 5 Epochs (sec)", fontsize=18)

    plt.xticks(y_pos, bars, fontsize=14)

    plt.savefig("./../results/nodeTimes.png")

def printCombinationTimes():
    data = [[282.1651, 297.4415, 293.7445],
            [283.5931, 298.6039, 293.3214]]
    X = np.arange(3)
    plt.figure(figsize=(10,10))
    plt.bar(X + 0.00, data[0],  width=0.25, label="average")
    plt.bar(X + 0.25, data[1], width=0.25, label="weighted")

    bars = ("Random", "Equal", "Class")
    y_pos = [(x+0.12) for x in range(len(bars))]
    plt.xticks(y_pos, bars, fontsize=18)

    plt.legend(fontsize=14)

    plt.savefig("./../results/combinationTimes.png")


if __name__ == "__main__":
    printNodeTimes()