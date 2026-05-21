import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from matplotlib.pyplot import figure

points = {"blue":[[2,4,3],[1,3,5],[2,3,1],[3,2,3],[2,1,6]],
          "red":[[5,6,5],[4,5,2],[4,6,1],[6,6,1],[5,4,6],[10,10,4]]}

new_points = [3,3,4]

def euclidean_distance(p,q):
    return np.sqrt(np.sum((np.array(p) - np.array(q))**2))

class KNearestNeighbors:
    def __init__(self,k=3):
        self.k = k
        self.points = None

    def fit(self,points):
        self.points = points

    def predict(self,new_point):
        distances = []

        for category in self.points:
            for point in self.points[category]:
                distance = euclidean_distance(point, new_point)
                distances.append([distance, category])

        category = [c[1] for c in sorted(distances,key=lambda  x:x[0])[:self.k]]
        result = Counter(category).most_common(1)[0][0]

        return result

clf = KNearestNeighbors()
clf.fit(points)

print(clf.predict(new_points))



ax = plt.figure(figsize=(15,12)).add_subplot(projection="3d")
ax.grid(True,color='grey')
ax.figure.set_facecolor("#121212")
ax.tick_params(axis="x",color="white")
ax.tick_params(axis="y",color="white")

for point in points['blue']:
    ax.scatter(point[0],point[1],point[2],color='#104DCA',s=60)

for point in points['red']:
    ax.scatter(point[0], point[1],point[2], color='#FF0000', s=60)

new_class = clf.predict(new_points)
color = "#FF0000" if new_class == "red" else "#104DCA"
ax.scatter(new_points[0],new_points[1],new_points[2],color = color,marker="*",s=200,zorder=100)

for point in points['blue']:
    ax.plot([new_points[0],point[0]],[new_points[1],point[1]],[new_points[2],point[2]],color='#104DCA',linestyle="--",linewidth=1)

for point in points['red']:
    ax.plot([new_points[0], point[0]], [new_points[1], point[1]],[new_points[2],point[2]], color='#FF0000', linestyle="--", linewidth=1)

plt.show()
