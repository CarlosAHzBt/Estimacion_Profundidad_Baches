#Clase para cargar el modelo de segmentación
from RutaModelo.Segformer_FineTuner import SegformerFinetuner
import torch

class CargarModelo:
        
    def cargar_modelo(self,ruta_modelo):
        ruta_modelo = ruta_modelo
     # Crear una nueva instancia del modelo
        id2label = {
            0: "background",
            1: "Bache",
            2: "Grieta",
            }
        modelo = SegformerFinetuner(id2label=id2label)  # Asegúrate de proporcionar los argumentos necesarios aquí

     # Cargar el state_dict guardado en el modelo
        modelo.load_state_dict(torch.load(ruta_modelo))

     # Cambiar el modelo a modo de evaluación si se va a hacer inferencia
        modelo.eval()

        return modelo
