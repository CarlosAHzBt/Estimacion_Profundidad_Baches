import numpy as np
import math
from ObtenerAlturaDeCaptura import AlturaCaptura

class ConvertirPixelesAMetros:
    def __init__(self):
        # Valores estándar para el campo de visión y resolución
        self.fov_horizontal = 69  # FoV horizontal en grados
        self.fov_vertical = 42   # FoV vertical en grados
        self.resolucion_ancho = 848 # Resolución en píxeles (ancho) 
        self.resolucion_alto = 480  # Resolución en píxeles (alto)

    def estimar_altura_de_captura(self,ply_path):
        """
        Estima la altura de captura de la nube de puntos PLY.
        """
        altura_captura = AlturaCaptura(ply_path)
        return altura_captura.calcular_altura()

    
    def calcular_escala(self, altura_captura):
        """
        Calcula las escalas de conversión de píxeles a metros basadas en la altura de captura.
        """
        ancho_real = 2 * altura_captura * math.tan(math.radians(self.fov_horizontal / 2))
        alto_real = 2 * altura_captura * math.tan(math.radians(self.fov_vertical / 2))
        escala_horizontal = ancho_real / self.resolucion_ancho
        escala_vertical = alto_real / self.resolucion_alto
        return escala_horizontal, escala_vertical

    def convertir_radio_pixeles_a_metros(self, radio_pixeles, escala_horizontal):
        """
        Convierte el radio del círculo de píxeles a metros usando la escala horizontal.
        """
        radio_metros = radio_pixeles * escala_horizontal
        return radio_metros

   
