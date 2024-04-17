#Script que Extraer la imagenes de profunidadd y RGB 
#Aplica un modelo de deteccion de baches.
#Guarda las coordenadas en una variable de la deteccion
#Genera las coordenadas como un bounding box
#Aplica las coordenadas sobre la imagen de profundidad
#Recorta la imagen de profundidad con las coordenadas manteniendo solo el interior del bounding box
#Genera una nube de puntos.

from LogicaExtraccionBag import ProcesadorBags
import numpy as np
import os
from CargarModelo import CargarModelo
from ModeloSegmentacion import ModeloSegmentacion
from Bache import Bache
from AdministradorDeArchivos import AdministradorArchivos
from LogicaExtraccionBag.ProcesadorBags import ProcesadorBags
import torch
import csv


class Main:
    def __init__(self, path_bag_folder):
        self.path_bag_folder = path_bag_folder
        self.modelo = None
        self.lista_baches = []

    def extraccion_informacion(self):
        extraccion_bag = ProcesadorBags(self.path_bag_folder)
        extraccion_bag.process_bag_files("Extraccion")

    def cargar_modelo(self):
        modelo_loader = CargarModelo()
        self.modelo = modelo_loader.cargar_modelo("RutaModelo/model_state_dictV5.pth")
        if torch.cuda.is_available():
            self.modelo.to('cuda')
        else:
            print("CUDA no está disponible, el modelo se ejecutará en CPU.")

    def aplicar_modelo(self):
        segmentador = ModeloSegmentacion(self.modelo)
        administrador_archivos = AdministradorArchivos("Extraccion")
        archivos_bags = administrador_archivos.generar_lista_de_archivosBags()
        for ruta_carpeta_bag in archivos_bags:
            imagenes = administrador_archivos.generar_lista_de_imagenes(ruta_carpeta_bag)
            for ruta_imagen in imagenes:
                coordenadas_baches = segmentador.obtener_coordenadas_baches(ruta_imagen)
                for i, coord in enumerate(coordenadas_baches):
                    id_bache = f"{os.path.splitext(os.path.basename(ruta_imagen))[0]}_{i}"
                    bag_de_origen = administrador_archivos.obtener_bag_de_origen(ruta_imagen)
                    bache = Bache(ruta_carpeta_bag, bag_de_origen, ruta_imagen, id_bache, coord)
                    if bache.procesar_bache() is True:
                        break
                    print(f"El diametro máximo del bache {bache.id_bache} es {bache.diametro_bache} mm procedente del bag {bache.bag_de_origen}.")
                    self.lista_baches.append(bache)

    def aplicar_recorte_a_imagenes_que_contengan_bache(self):
        for bache in self.lista_baches:
            #bache.recortar_y_procesar_nube_de_puntos()
            #profundidad = bache.estimar_profundidad_del_bache()
            print(f" La profundidad del bache {bache.id_bache} es de {bache.profundidad_del_bache_estimada} m.")

    def borrar_todos_los_archivos_extraidos_al_terminar(self):
        administrador_archivos = AdministradorArchivos("Extraccion")
        administrador_archivos.borrar_archivos_extraidos()

    def generar_documento_de_deterioros(self):
        # Genera un documento con la información de los baches detectados y sus características en formato CSV
        with open('deterioros.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["ID Bache", "Radio Máximo (mm)", "Profundidad (m)", "Imagen"])
            for bache in self.lista_baches:
                # Genera la imagen con contorno y círculo
                writer.writerow([bache.id_bache, bache.diametro_bache, bache.profundidad_del_bache_estimada, bache.ruta_imagen_contorno])

    def run(self):
        #self.extraccion_informacion()
        self.cargar_modelo()
        self.aplicar_modelo()
        self.aplicar_recorte_a_imagenes_que_contengan_bache()
        self.generar_documento_de_deterioros()
        #self.borrar_todos_los_archivos_extraidos_al_terminar()

if __name__ == "__main__":
    path_bag_folder = "bag"
    app = Main(path_bag_folder)
    app.run()
