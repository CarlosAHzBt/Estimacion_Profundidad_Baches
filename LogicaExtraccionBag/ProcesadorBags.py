#Clase que procesa multiples bags
from LogicaExtraccionBag.BagFile import BagFile
import os


class ProcesadorBags:
    def __init__(self, bag_files_path):
        self.bag_files_path = bag_files_path
        self.bag_files = self.get_bag_files()

    def get_bag_files(self):
        """
        Obtiene la lista de archivos .bag en el directorio especificado.
        """
        return [f for f in os.listdir(self.bag_files_path) if f.endswith('.bag')]

    def process_bag_files(self,carpeta_destino):
        """
        Procesa todos los archivos .bag en el directorio especificado.
        """
        for bag_file in self.bag_files:
            bag_file_path = f"{self.bag_files_path}/{bag_file}"
            bag = BagFile(bag_file_path,carpeta_destino)
            bag.process_bag_file()
        print("Bags Procesados---------------------------------")


    def process_bag_file(self, bag_file_path,carpeta_destino):
        """
        Procesa un archivo .bag.
        """
        bag_file_path = f"{self.bag_files_path}/{bag_file_path}"
        bag = BagFile(bag_file_path,carpeta_destino)
        bag.process_bag_file()