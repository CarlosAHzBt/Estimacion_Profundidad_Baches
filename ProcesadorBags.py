from LogicaExtraccionBag.BagFile import BagFile
import os
import logging
import concurrent.futures

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ProcesadorBags:
    def __init__(self, bag_files_path):
        self.bag_files_path = bag_files_path
        self.bag_files = self.get_bag_files()

    def get_bag_files(self):
        """
        Obtiene la lista de archivos .bag en el directorio especificado.
        """
        try:
            bag_files = [f for f in os.listdir(self.bag_files_path) if f.endswith('.bag')]
            if not bag_files:
                logging.warning("No se encontraron archivos .bag en el directorio.")
            return bag_files
        except FileNotFoundError:
            logging.error(f"El directorio {self.bag_files_path} no existe.")
            return []

    def process_bag_files(self, carpeta_destino):
        """
        Procesa todos los archivos .bag en el directorio especificado de forma concurrente.
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.process_bag_file, bag_file, carpeta_destino): bag_file for bag_file in self.bag_files}
            for future in concurrent.futures.as_completed(futures):
                bag_file = futures[future]
                try:
                    future.result()  # Aquí podrías manejar resultados específicos si es necesario
                except Exception as exc:
                    logging.error(f'El archivo {bag_file} generó un error: {exc}')

        logging.info("Todos los archivos .bag han sido procesados.")

    def process_bag_file(self, bag_file, carpeta_destino):
        """
        Procesa un archivo .bag.
        """
        try:
            bag_file_path = os.path.join(self.bag_files_path, bag_file)
            bag = BagFile(bag_file_path, carpeta_destino)
            bag.process_bag_file()
        except Exception as e:
            logging.error(f"Error al procesar el archivo {bag_file}: {e}")
