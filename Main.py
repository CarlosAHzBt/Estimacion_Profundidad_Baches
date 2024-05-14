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
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Main:
    def __init__(self, path_bag_folder, output_folder, batch_size=1):
        self.path_bag_folder = path_bag_folder
        self.output_folder = output_folder
        self.modelo = None
        self.ruta_modelo = "model_state_dictV18-este_ya_trae_ruido.pth"  # Ruta del modelo entrenado
        self.lista_baches = []
        self.batch_size = batch_size

    def run(self):
        self.extraccion_informacion()
        self.cargar_modelo()
        self.aplicar_modelo()
        self.procesar_baches()
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
        
        # Crear una lista de todas las imágenes para asegurar el procesamiento completo
        lista_imagenes = []
        for ruta_carpeta_bag in archivos_bags:
            imagenes = administrador_archivos.generar_lista_de_imagenes(ruta_carpeta_bag)
            for ruta_imagen in imagenes:
                lista_imagenes.append((ruta_carpeta_bag, ruta_imagen))

        # Procesar imágenes en lotes
        for i in range(0, len(lista_imagenes), self.batch_size):
            batch = lista_imagenes[i:i+self.batch_size]
            self.procesar_lote_imagenes(segmentador, administrador_archivos, batch)

    def procesar_lote_imagenes(self, segmentador, administrador_archivos, batch):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        segmentador.modelo.to(device)  # Asegúrate de que el modelo esté en la GPU
        with ThreadPoolExecutor() as executor:
            future_to_image = {executor.submit(self.procesar_imagen, segmentador, administrador_archivos, ruta_carpeta_bag, ruta_imagen, device): (ruta_carpeta_bag, ruta_imagen) for ruta_carpeta_bag, ruta_imagen in batch}
            for future in as_completed(future_to_image):
                try:
                    baches = future.result()
                    self.lista_baches.extend(baches)
                    
                except Exception as exc:
                    ruta_carpeta_bag, ruta_imagen = future_to_image[future]
                    logging.error(f"Error procesando la imagen {ruta_imagen} en la carpeta {ruta_carpeta_bag}: {exc}")
        torch.cuda.empty_cache()  # Liberar memoria de la GPU después de procesar el lote

    def procesar_imagen(self, segmentador, administrador_archivos, ruta_carpeta_bag, ruta_imagen, device):
        # Obtener las coordenadas de los baches
        coordenadas_baches = segmentador.obtener_coordenadas_baches(ruta_imagen)
        
        baches = []
        for i, coord in enumerate(coordenadas_baches):
            id_bache = f"{os.path.splitext(os.path.basename(ruta_imagen))[0]}_{i}"
            bag_de_origen = administrador_archivos.obtener_bag_de_origen(ruta_imagen, self.path_bag_folder)
            bache = Bache(ruta_carpeta_bag, bag_de_origen, self.output_folder, ruta_imagen, id_bache, coord)
            baches.append(bache)
        return baches

    def procesar_baches(self):
        logging.info("Procesando baches identificados.")
        nueva_lista_baches = []
        with ThreadPoolExecutor() as executor:
            future_to_bache = {executor.submit(self.procesar_bache, bache): bache for bache in self.lista_baches}
            for future in as_completed(future_to_bache):
                try:
                    bache = future.result()
                    if bache is not None:
                        nueva_lista_baches.append(bache)
                except Exception as exc:
                    logging.error(f"Error procesando el bache: {exc}")
        self.lista_baches = nueva_lista_baches

    def procesar_bache(self, bache):
        if bache.procesar_bache():
            logging.info(f"El diámetro máximo del bache {bache.id_bache} es {bache.diametro_bache} mm procedente del bag {bache.bag_de_origen}.")
            if bache.profundidad_del_bache_estimada < -0.015:  # Si la profundidad del bache es menor a 15 mm no se toma en cuenta
                logging.info(f"Bache {bache.id_bache} con profundidad {bache.profundidad_del_bache_estimada} m agregado a la lista final.")
                return bache
        return None

    def borrar_todos_los_archivos_extraidos_al_terminar(self):
        logging.info("Borrando todos los archivos extraídos al finalizar el proceso.")
        administrador_archivos = AdministradorArchivos("Extraccion")
        administrador_archivos.borrar_archivos_extraidos()

    def generar_documento_de_deterioros(self):
        logging.info("Generando documento de registro de deterioros.")
        # Ordenar la lista de baches por ID y bag de origen
        self.lista_baches = sorted(self.lista_baches, key=lambda bache: (bache.bag_de_origen, bache.id_bache))
        with open(os.path.join(self.output_folder, 'deterioros.csv'), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["ID Bache", "Radio Máximo (mm)", "Profundidad (m)", "Imagen"])
            for bache in self.lista_baches:
                writer.writerow([bache.id_bache, bache.diametro_bache, bache.profundidad_del_bache_estimada, bache.ruta_imagen_contorno])
