from LogicaExtraccionBag.ProcesadorBags import ProcesadorBags
import os
from CargarModelo import CargarModelo
from ModeloSegmentacion import ModeloSegmentacion
from Bache import Bache
from AdministradorDeArchivos import AdministradorArchivos
import torch
import csv
import logging
import sys

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Main:
    def __init__(self, path_bag_folder, output_folder):
        self.path_bag_folder = path_bag_folder
        self.output_folder = output_folder
        self.modelo = None
        self.ruta_modelo = "model_state_dictV13.pth"  # Ruta del modelo entrenado
        self.lista_baches = []

    def run(self):
        self.extraccion_informacion()
        self.cargar_modelo()
        self.aplicar_modelo()
        self.procesar_baches()
        self.aplicar_recorte_a_imagenes_que_contengan_bache()
        self.generar_documento_de_deterioros()
        self.borrar_todos_los_archivos_extraidos_al_terminar()

    def extraccion_informacion(self):
        logging.info("Iniciando la extracción de información desde archivos bag.")
        extraccion_bag = ProcesadorBags(self.path_bag_folder)
        extraccion_bag.process_bag_files("Extraccion")

    def cargar_modelo(self):
        logging.info("Cargando modelo de segmentación.")
        modelo_loader = CargarModelo()
        if getattr(sys, 'frozen', False):
            # Si se ejecuta como ejecutable congelado, la ruta es relativa al sys._MEIPASS
            ruta_modelo = os.path.join(sys._MEIPASS, self.ruta_modelo)
        else:
            # Ruta normal como script de Python
            ruta_modelo = self.ruta_modelo
        self.modelo = modelo_loader.cargar_modelo(ruta_modelo)
        self.modelo.to('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info("Modelo cargado y transferido a " + ('CUDA' if torch.cuda.is_available() else 'CPU'))

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
                    bag_de_origen = administrador_archivos.obtener_bag_de_origen(ruta_imagen, self.path_bag_folder)
                    bache = Bache(ruta_carpeta_bag, bag_de_origen, self.output_folder, ruta_imagen, id_bache, coord)
                    self.lista_baches.append(bache)
                    logging.info(f"Bache {id_bache} identificado en la imagen {ruta_imagen}.")

    def procesar_baches(self):
        logging.info("Procesando baches identificados.")
        for bache in self.lista_baches:
            if bache.procesar_bache():
                logging.info(f"El diámetro máximo del bache {bache.id_bache} es {bache.diametro_bache} mm procedente del bag {bache.bag_de_origen}.")
                if bache.profundidad_del_bache_estimada < -0.01:
                    logging.info(f"Bache {bache.id_bache} con profundidad {bache.profundidad_del_bache_estimada} m agregado a la lista final.")

    def aplicar_recorte_a_imagenes_que_contengan_bache(self):
        logging.info("Aplicando recorte a imágenes que contengan baches y procesando nubes de puntos.")
        for bache in self.lista_baches:
            logging.info(f"La profundidad del bache {bache.id_bache} es de {bache.profundidad_del_bache_estimada} m.")

    def borrar_todos_los_archivos_extraidos_al_terminar(self):
        logging.info("Borrando todos los archivos extraídos al finalizar el proceso.")
        administrador_archivos = AdministradorArchivos("Extraccion")
        administrador_archivos.borrar_archivos_extraidos()

    def generar_documento_de_deterioros(self):
        logging.info("Generando documento de registro de deterioros.")
        with open(os.path.join(self.output_folder, 'deterioros.csv'), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["ID Bache", "Radio Máximo (mm)", "Profundidad (m)", "Imagen"])
            for bache in self.lista_baches:
                writer.writerow([bache.id_bache, bache.diametro_bache, bache.profundidad_del_bache_estimada, bache.ruta_imagen_contorno])

