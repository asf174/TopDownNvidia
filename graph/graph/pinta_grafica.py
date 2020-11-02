import sys
import pathlib
import os.path
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
	print("The arguments passed: ", str(sys.argv), "are incorrect")
else:
	directorio = pathlib.Path.cwd().joinpath('ficheros', sys.argv[len(sys.argv)-1])
	if(os.path.isfile(directorio)):
		print("Plotting data from file: ", directorio)
		fich = open (directorio,'r')
		array_x = []
		array_y = []
		for line in fich.readlines():
			media = 0
			contador = 0
			bifurcacion = line.strip().split(":")
			array_x.append(int(bifurcacion[0]))
			for add in bifurcacion[1].split():
				media = media + int(add)
				contador = contador + 1
			array_y.append(media/contador)
		fich.close()
		print("Datos: ", array_y)
		print("Ejecucion: ", array_x)

		#Once data has been collected, plotting must be done
		numero_graficas = 2
		fig, axs = plt.subplots(1, numero_graficas)
		axs[0].plot(array_x, array_y)
		axs[1].plot(array_x, array_y)
		fig.suptitle(sys.argv[-1])
		for ax in range(0, numero_graficas):
			axs[ax].set(xlabel="x - Iteration", ylabel="y - Data")
		for ax in axs.flat:
    			ax.label_outer()
		axs[0].set_title("Plot")
		axs[1].set_title("Plot")
		axs[0].grid()
		axs[1].grid()
		plt.show()

	else:
		print("Unknown destination / not a file")

