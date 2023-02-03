import os
import matplotlib.pyplot as plt
import json


class PlotPie:
    def __init__(self, pathToJson):
        self.pathToJson = pathToJson
        self.data = self.read_data()

    def sort_dict_by_values(self, d):
        return dict(sorted(d.items(), key=lambda item: item[1], reverse=True))

    def read_data(self):
        data=[]
        data = json.load(open(self.pathToJson))
        for d in data :

            if len(d["Paternity"]) == 0:
                data.remove(d)
        return data

    def plot(self):
        i = 0
        fig, axes = plt.subplots(1, len(self.data), figsize=(5 * len(self.data), 5))
        fig.subplots_adjust(wspace=0.5, hspace=0.5)
        fig.suptitle("Paternity Per Type of Variability", fontsize=16)

        for Variabilty in self.data:
            i += 1
            VariabilityType = Variabilty["VariabilityType"]
            Paternity = Variabilty["Paternity"]
            if len(Paternity)==0 :
                break

            explode = [0 for element in range(len(Paternity))]
            explode[0] = 0.1
            Paternity = self.sort_dict_by_values(Paternity)

            axes[i - 1].pie(Paternity.values(), labels=Paternity.keys(), explode=explode, shadow=True, autopct='%1.2f%%')
            axes[i - 1].set_title(VariabilityType)

        file_name, file_extension = os.path.splitext(self.pathToJson)
        if not os.path.isdir("Visualization/"+file_name): os.makedirs("Visualization/"+file_name)
        plt.savefig("Visualization/"+file_name+"/MultiPie.png")
        plt.show()


if __name__ == '__main__':
    plot_pie = PlotPie("Visualization/input/jafka_paternity_result.json")
    plot_pie.plot()