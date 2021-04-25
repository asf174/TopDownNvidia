import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class PieChart:
    """
    Class which defines a PieChart graph.

    Attributes:
        __fig       : fig   ; reference to diagram (which contains all graphs)
        __max_rows  : int   ; max number of rows
        __max_cols  : int   ; max number of cols
        __numCols   : int   ; current number of cols added to diagram
        __numRows   : int   ; current numer of rows added to diagram
        __title     : str   ; title name of diagram
    """

    def __init__(self, rows : int, cols : int, title : str, titles_sub_graphs : list):
        """Set attributes as arguments."""
        specs_l : list[list] = list(list())
        specs_sl : list = list()
        for i in range (0, rows):
            for j in range(0, cols):
                specs_sl.append({'type' : 'domain'})
            specs_l.append(specs_sl)
        self.__fig = make_subplots(rows = rows, cols = cols, specs = specs_l, subplot_titles = titles_sub_graphs)
        self.__max_rows : int = rows
        self.__max_cols : int = cols
        self.__num_cols : int = 0
        self.__num_rows : int = 1
        self.__title : str = title
        pass


    def draw(labels : list, sizes : list, explode : list):
        if len(labels) != len(sizes) or len(labels) != len(explode):
             return False
        plt.pie(sizes, explode = explode, labels = labels, autopct='%1.1f%%',
            shadow = True, startangle = 90)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.savefig('books_read.png')
        pass

    def add_graph(self, labels : list, values : list, name : str, legend_group : str) -> bool:
        if self.__num_cols > self.__max_cols or self.__num_rows > self.__max_rows:
            return False
        self.__num_cols += 1
        if self.__num_cols > self.__max_cols:
            self.__num_cols = 1
            self.__num_rows += 1
            if self.__num_rows > self.__max_rows:
                return False
        print(self.__num_rows)
        print(self.__num_cols)
        self.__fig.add_trace(go.Pie(labels = labels, values = values, name = name, showlegend = True, legendgroup = legend_group), row = self.__num_rows, col = self.__num_cols)
        self.__fig.update_yaxes(title_text = name, row = self.__num_rows, col = self.__num_rows)
        return True
        pass
    
    def print(self):
        plt.tight_layout()
        self.__fig.update_layout(title = {'text' : self.__title, 'x' : 0.5, 'xanchor': 'center'}, legend = dict(yanchor = "top", y = 0.99, xanchor = "left", x = 0.01), legend_title = "Legend", font = dict(size = 12, color = "Black"))
        self.__fig.show()
        pass
        
