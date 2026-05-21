import numpy as np
import matplotlib.pyplot as plt
from collections import  Counter

points = {"blue":[[2,4],[1,3],[2,3],[3,2],[2,1]],
          "red":[[5,6],[4,5],[4,6],[6,6],[5,4]]}

new_points = [5,3]
k=3

def euclidian_distance(p,q):
    return np.sqrt(np.sum((np.array(p)-np.array(q))**2))


class KNN:
    def __init__(self,k):
        self.k = k
        self.points = None

    def fit(self,points):
        self.points = points

    def predict(self,new_points):
        distance = []
        for category in self.points:
            for point in self.points[category]:
                distance.append([euclidian_distance(point,new_points),category])

        return Counter([x[1] for x in sorted(distance)[:self.k]]).most_common(1)[0][0]


clf = KNN(k=3)
clf.fit(points)
print(clf.predict(new_points))


for point in points["blue"]:
    plt.scatter(point[0],point[1],color="blue",s=60)

for point in points["red"]:
    plt.scatter(point[0],point[1],color="red",s=60)



for point in points["blue"]:
    plt.plot([new_points[0],point[0]],[new_points[1],point[1]],color="blue")

for point in points["red"]:
    plt.plot([new_points[0],point[0]],[new_points[1],point[1]],color="red")



plt.show()
