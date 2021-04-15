import matplotlib.pyplot as plt

class PieChart:

    def draw(labels : lst, sizes : lst, explode : lst):
        if len(labels) != len(sizes) or len(labels) != len(explode):
             return False
        plt.pie(sizes, explode = explode, labels = labels, autopct='%1.1f%%',
            shadow = True, startangle = 90)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.savefig('books_read.png')
        pass

        
