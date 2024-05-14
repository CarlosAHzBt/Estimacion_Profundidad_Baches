import threading
import os
from LogicaExtraccionBag.BagFile import BagFile

class ProcesadorBags:
    def __init__(self, bag_files_path):
        self.bag_files_path = bag_files_path
        self.bag_files = self.get_bag_files()
        self.lock = threading.Lock()  # Lock para sincronización

    def get_bag_files(self):
        """Obtiene la lista de archivos .bag en el directorio especificado."""
        return [f for f in os.listdir(self.bag_files_path) if f.endswith('.bag')]

    def process_bag_files(self, carpeta_destino):
        """Procesa todos los archivos .bag en el directorio especificado usando hilos."""
        threads = []
        for bag_file in self.bag_files:
            thread = threading.Thread(target=self.process_bag_file, args=(bag_file, carpeta_destino))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()  # Espera a que todos los hilos terminen
        print("Bags procesados ---------------------------------")

    def process_bag_file(self, bag_file, carpeta_destino):
        """Procesa un archivo .bag."""
        bag_file_path = f"{self.bag_files_path}/{bag_file}"
        bag = BagFile(bag_file_path, carpeta_destino, self.lock)
        bag.process_bag_file()