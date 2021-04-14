import matplotlib.pyplot as plt

class PieChart:

    def print(labels : lst, sizes : lst, explode : lst):
        if len(labels) != len(sizes) or len(labels) != len(explode):
            return
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        #plt.show()
        plt.savefig('books_read.png')
