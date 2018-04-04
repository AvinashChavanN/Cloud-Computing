from flask import Flask, render_template, request, redirect
import pygal
from numpy import vstack
from scipy.cluster.vq import kmeans, vq

app = Flask(__name__)


@app.route("/")
def IndexPage():
    return render_template('index.html')

@app.route("/select",methods=['GET', 'POST'])
def Show_Graph():
    if request.method == "POST":
        with open('titanic.csv', 'r') as Input:
            AgevsClass = []
            AgeVsFare_Data = []
            AgeVsSurvived_Data = []
            FareVsSurvived_Data = []
            PClassVsFare_Data = []
            Input.readline()  # Skip the header
            Data = Input.readline()  # Read 2nd line from CSV
            for line in Input:
                Data = line.split(',')  # Create a list of the data
                PClass = float(Data[0])
                Survived = float(Data[1])
                if Data[2].isdigit():
                    Age = float(Data[2])
                else:
                    Age = 0.0
                if Data[3].isdigit():
                    Fare = float(Data[3].rstrip('\r\n'))
                else:
                    Fare = 0.0
                AgevsClass.append((Age, PClass))
                AgeVsFare_Data.append((Age, Fare))
                AgeVsSurvived_Data.append((Age, Survived))
                FareVsSurvived_Data.append((Survived, Fare))
                PClassVsFare_Data.append((PClass, Fare))
        ClusterData = vstack(FareVsSurvived_Data)  # Change input according to graph plot
        input1 = request.form["ClusterNumber"]
        ClusterNumber = int(input1)
        # computing K-Means with K = 2 (2 clusters)
        centroids, _ = kmeans(ClusterData, ClusterNumber)
        # assign each sample to a cluster
        xy_chart = pygal.XY(stroke=False)
        xy_chart.title = 'Clustering'
        xy_chart.x_title = 'Survived'
        xy_chart.y_title = 'Fare'
        idx, _ = vq(ClusterData, centroids)
        for num in range(0, ClusterNumber):
            xDataSize = ClusterData[idx == num, 0].size
            yDataSize = ClusterData[idx == num, 1].size
            xDataArray = list()
            yDataArray = list()
            xyDataElements = []
            for i in ClusterData[idx == num, 0]:
                xDataArray.append(i)
            for i in ClusterData[idx == num, 1]:
                yDataArray.append(i)
            for i in range(0, xDataSize):
                xyDataElements.append((float(xDataArray[i]), float(yDataArray[i])))
            s = "cluster" + str(num)
            xy_chart.add(s, xyDataElements)
        xClusterArray = list()
        yClusterArray = list()
        xyCusterElements = []
        x = centroids[:, 0].size
        for i in centroids[:, 0]:
            xClusterArray.append(i)
        for i in centroids[:, 1]:
            yClusterArray.append(i)
        for i in range(0, x):
            xyCusterElements.append((float(xClusterArray[i]), float(yClusterArray[i])))
        xy_chart.add('Centroid', xyCusterElements)
        return xy_chart.render_response()

if __name__ == "__main__":
    app.run(debug=True)