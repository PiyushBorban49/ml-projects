import random
from matplotlib import style
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import lineStyles

style.use("ggplot")

#scatters data

#c = color of points
#markers = points shape
#s = size of points
#alpha helps in making the points transparent that are being overlaped due to large dataset

# X_data = np.random.random(500)*100
# Y_data = np.random.random(500)*100

# plt.scatter(X_data,Y_data,c="red",marker="*",s=150,alpha=0.3)
# plt.show()





#plotting data as interconnected lines

#c = color of line
#lw = width of the line
#linestyle = style of the line

# years = [2000+x for x in range(25)]
# weights = []
# for i in range(25):
#     weights.append(random.randint(40,100))
#
# plt.plot(years,weights,c="red",lw=1,linestyle="--")
# plt.show()




#bar chart

# X = ["A","B","C","D","E"]
# Y = []
# for i in range(5):
#     Y.append(random.randint(20,140))
#
# plt.bar(X,Y,align="center",color="red",width=0.5,edgecolor="black",lw=3)
# plt.show()



#histogram chart

#loc = mean
#scale = standard deviation
#bins = number of bins

# ages = np.random.normal(20,7.5,1000)
# plt.hist(ages,bins=50,cumulative=True)
# plt.show()




#pie chart

#explode slides the chossen pie part outwords
#autopct = helps in representing percentage of that label
#pctdistance = helps in representing percentage label at a certain distance from center
#startangle = helps in defining the starting point of the pie chart

# langs = ["A","B","C","D","E"]
# votes = [50,24,14,6,12]
# plt.pie(votes,labels=langs,explode=[0,0,0,0.2,0],autopct="%.2f%%",pctdistance=1.25,startangle=90)
# plt.show()






#box plot

# heights = np.random.normal(172,8,300)
# plt.boxplot(heights)
# plt.show()





#Example of plot customization

#1.plot labeling

# years = [i for i in range(2016,2025)]
# income = []
# for i in range(0,len(years)):
#     income.append(random.randint(100,1000))
#
# plt.plot(years,income)
# plt.title("This is a year vs income chart",fontsize = 25,fontname="Broadway")
# plt.xlabel("Year")
# plt.ylabel("Yearly Income")
# plt.show()


#2.plot legends

# stock_a = [100,102,99,101,100,102]
# stock_b = [90,95,102,104,105,103,109]
# stock_c = [110,115,100,105,100,98,95]
#
# plt.plot(stock_a,label="Company1")
# plt.plot(stock_b,label="Company2")
# plt.plot(stock_c,label="Company3")
#
# plt.legend(loc="upper right")
#
# plt.show()


#Multiple figure

# x,y = np.random.random(100),np.random.random(100)
# s,t = np.arange(100),np.random.random(100)
#
# plt.figure(1)
# plt.scatter(x,y)
#
# plt.figure(2)
# plt.plot(s,t)
#
# plt.show()




#Subplotting

# x = np.arange(100)
#
# fig , axis = plt.subplots(2,2)
#
# axis[0,0].plot(x,np.sin(x))
# axis[0,0].set_title("Sin Wave")
#
# axis[0,1].plot(x,np.cos(x))
# axis[0,1].set_title("Cos Wave")
#
# axis[1,0].scatter(np.random.random(100),np.random.random(100))
# axis[1,0].set_title("Lines")
#
# axis[1,1].plot(x,np.log(x))
# axis[1,1].set_title("Log Wave")
#
# plt.show()
#
#
# plt.savefig("test.png",dpi=300)




#3D plotting


#1.Scatter Plotting
# axis = plt.axes(projection="3d")
#
# X=np.random.random(100)
# Y=np.random.random(100)
# Z=np.random.random(100)
#
# axis.scatter(X,Y,Z)
# axis.set_title("3D Plotting")
# plt.show()


#2.Line Plotting
# axis = plt.axes(projection="3d")
#
# X=np.arange(0,50,0.1)
# Y=np.arange(0,50,0.1)
# Z=np.cos(X)
#
# axis.plot(X,Y,Z)
# axis.set_title("3D Plotting")
# plt.show()





#Area Plotting
# axis = plt.axes(projection="3d")
#
# X=np.arange(-5,5,0.1)
# Y=np.arange(-5,5,0.1)
# X,Y = np.meshgrid(X,Y)
#
# Z = np.sin(X)*np.cos(Y)
#
# axis.plot_surface(X,Y,Z,cmap = "Spectral")
# axis.view_init(azim=0,elev=90)
# plt.show()





#Animation
heads_tails = [0,0]
for _ in range(10000):
    heads_tails[random.randint(0,1)] +=1
    plt.bar(["Heads","Tails"],heads_tails,color=["red","blue"])
    plt.pause(0.001)

plt.show()