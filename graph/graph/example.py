import matplotlib.pyplot as plt

def traza_curvas(series, colores, nombres):
    ''' Traza varias curvas a partir de varias series de puntos

    Además de la serie de puntos, se toma como entrada una serie de
    etiquetas y una serie de colores. Se debe comprobar que las tres
    series tienen el mismo número de elementos.
    Puede ser de utilidad la función 'built-in' zip
    '''
    if len(series) != len(colores) or len(series) != len(nombres):
        raise ValueError("Las tres series deben tener el mismo tamaño")
    for serie, color, nombre in zip(series, colores, nombres):
        plt.plot(serie, label=nombre, color=color)
    plt.legend()
    plt.show()


# Test de la función traza_curvas
series = [[0, 4,  5,  7,  8,  9, 10],
          [0, 1,  1.5,  2,  3,  7,  7.5], 
          [0,  5,  7,  8,  14,  14.5,  16], 
          [0,  0.5,  1,  1.5,  2,  2.5,  3]]
colores = ['blue', 'red', 'orange', 'grey']
nombres = [ "Serie 1", "Serie 2", "Serie 3", "Serie 4"]
traza_curvas(series, colores, nombres)