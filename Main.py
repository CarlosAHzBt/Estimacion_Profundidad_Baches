from LogicaExtraccionBag.ProcesadorBags import ProcesadorBags
import os
from CargarModelo import CargarModelo
from ModeloSegmentacion import ModeloSegmentacion
from Bache import Bache
from AdministradorDeArchivos import AdministradorArchivos
from SeguimientoBaches import SeguimientoBaches
import torch
import csv
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
class Main:
    def __init__(self, path_bag_folder, output_folder, batch_size=20):
        self.path_bag_folder = path_bag_folder
        self.output_folder = output_folder
        self.modelo = None
        self.ruta_modelo = "model_state_dictV18-este_ya_trae_ruido.pth"
        self.lista_baches = []
        self.batch_size = batch_size
        self.seguimiento_baches = SeguimientoBaches()

    def run(self):
        self.extraccion_informacion()
        self.cargar_modelo()
        self.aplicar_modelo()
        self.procesar_baches()
        self.agrupar_baches_y_calcular_promedios()
        self.generar_documento_de_deterioros()
        #self.borrar_todos_los_archivos_extraidos_al_terminar()

    def extraccion_informacion(self):
        logging.info("Iniciando la extracción de información desde archivos bag.")
        extraccion_bag = ProcesadorBags(self.path_bag_folder)
        extraccion_bag.process_bag_files("Extraccion")

    def cargar_modelo(self):
        logging.info("Cargando modelo de segmentación.")
        modelo_loader = CargarModelo()
        if getattr(sys, 'frozen', False):
            ruta_modelo = os.path.join(sys._MEIPASS, self.ruta_modelo)
        else:
            ruta_modelo = self.ruta_modelo
        self.modelo = modelo_loader.cargar_modelo(ruta_modelo)
        self.modelo.to('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info("Modelo cargado y transferido a " + ('CUDA' if torch.cuda.is_available() else 'CPU'))

    def aplicar_modelo(self):
        segmentador = ModeloSegmentacion(self.modelo)
        administrador_archivos = AdministradorArchivos("Extraccion")
        archivos_bags = administrador_archivos.generar_lista_de_archivosBags()
        
        lista_imagenes = []
        for ruta_carpeta_bag in archivos_bags:
            imagenes = administrador_archivos.generar_lista_de_imagenes(ruta_carpeta_bag)
            for ruta_imagen in imagenes:
                lista_imagenes.append((ruta_carpeta_bag, ruta_imagen))

        for i in range(0, len(lista_imagenes), self.batch_size):
            batch = lista_imagenes[i:i+self.batch_size]
            self.procesar_lote_imagenes(segmentador, administrador_archivos, batch)


    def procesar_lote_imagenes(self, segmentador, administrador_archivos, batch):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        segmentador.modelo.to(device)
        with ThreadPoolExecutor() as executor:
            future_to_image = {}
            for frame_number, (ruta_carpeta_bag, ruta_imagen) in enumerate(batch):
                future = executor.submit(self.procesar_imagen, segmentador, administrador_archivos, ruta_carpeta_bag, ruta_imagen, device, frame_number)
                future_to_image[future] = (ruta_carpeta_bag, ruta_imagen, frame_number)
            
            for future in as_completed(future_to_image):
                try:
                    baches = future.result()
                    for bache in baches:
                        self.lista_baches.append(bache)
                except Exception as exc:
                    ruta_carpeta_bag, ruta_imagen, frame_number = future_to_image[future]
                    logging.error(f"Error procesando la imagen {ruta_imagen} en la carpeta {ruta_carpeta_bag}: {exc}")
        torch.cuda.empty_cache()

    def procesar_imagen(self, segmentador, administrador_archivos, ruta_carpeta_bag, ruta_imagen, device, frame_number):
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
            future_to_bache = {executor.submit(bache.procesar_bache): bache for bache in self.lista_baches}
            for future in as_completed(future_to_bache):
                try:
                    result = future.result()
                    if result:
                        nueva_lista_baches.append(future_to_bache[future])
                except Exception as exc:
                    logging.error(f"Error procesando el bache: {exc}")
                    self.lista_baches.remove(future_to_bache[future])
        self.lista_baches = nueva_lista_baches


    def agrupar_baches_y_calcular_promedios(self):
        logging.info("Agrupando baches y calculando promedios.")
        for bache in self.lista_baches:
            self.seguimiento_baches.agregar_bache(bache)
        agrupaciones = self.seguimiento_baches.agrupar_baches()
        self.lista_baches = self.seguimiento_baches.calcular_promedios(agrupaciones)

    def generar_documento_de_deterioros(self):
        logging.info("Generando documento de registro de deterioros.")
        self.lista_baches = sorted(self.lista_baches, key=lambda bache: (bache.bag_de_origen, bache.id_bache))
        baches_agrupados_ids = set()
        
        with open(os.path.join(self.output_folder, 'deterioros.csv'), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["ID Bache", "Radio Máximo Promedio (mm)", "Profundidad Promedio (m)", "Imagen", "Frames Agrupados"])
            
            for bache in self.lista_baches:
                if bache.id_bache not in baches_agrupados_ids:
                    # Agregar todos los frames agrupados al set para evitar duplicados
                    baches_agrupados_ids.update(bache.frames_agrupados)
                    # Escribir solo el primer bache de cada agrupación
                    writer.writerow([
                        bache.id_bache, 
                        bache.radio_maximo, 
                        bache.profundidad_del_bache_estimada, 
                        bache.ruta_imagen_contorno, 
                        ', '.join(bache.frames_agrupados)
                    ])


    def borrar_todos_los_archivos_extraidos_al_terminar(self):
        logging.info("Borrando todos los archivos extraídos al finalizar el proceso.")
        administrador_archivos = AdministradorArchivos("Extraccion")
        administrador_archivos.borrar_archivos_extraidos()