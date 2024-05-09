import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
from skimage.measure import label, regionprops
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt

class ModeloSegmentacion():
    def __init__(self, modelo_entrenado):
        self.min_area = 3000  # Área mínima para considerar una detección de bache válida
        self.modelo = modelo_entrenado
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _preparar_imagen(self, ruta_imagen):
        imagen = Image.open(ruta_imagen).convert("RGB")
        transformaciones = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        imagen_transformada = transformaciones(imagen).unsqueeze(0).to(self.device)
        return imagen_transformada

    def _aplicar_modelo(self, pixel_values):
        if torch.cuda.is_available():
            pixel_values = pixel_values.to('cuda')
        with torch.no_grad():
            predicciones = self.modelo(pixel_values=pixel_values)
            # Seleccionar solo la máscara de la clase 'Bache', asumiendo que la clase 'Bache' es 1
            predicted_mask = (predicciones[0].argmax(dim=1) == 1).squeeze().cpu().numpy().astype(int)
        return predicted_mask

    def _redimensionar_mascara(self, predicted_mask):
        predicted_mask_resized = resize(predicted_mask, (480, 848), order=0, 
                                        preserve_range=True, anti_aliasing=False).astype(int)
        return predicted_mask_resized

    def _etiquetar_regiones(self, mask_resized):
        labeled_mask = label(mask_resized, connectivity=2)
        return labeled_mask

    def _filtrar_regiones(self, labeled_mask):
        regions = regionprops(labeled_mask)
        filtered_regions = [region.coords for region in regions if region.area >= self.min_area]
        return filtered_regions
    
    #def _dibujar_regiones_filtradas(self, labeled_mask, filtered_regions):
    #    plt.imshow(labeled_mask, cmap="nipy_spectral")
    #    for region in filtered_regions:
    #        plt.plot(region[:, 1], region[:, 0], "o", markersize=3)
    #    plt.show()

    def obtener_coordenadas_baches(self, ruta_imagen):
        pixel_values = self._preparar_imagen(ruta_imagen)
        predicted_mask = self._aplicar_modelo(pixel_values)
        mask_resized = self._redimensionar_mascara(predicted_mask)
        labeled_mask = self._etiquetar_regiones(mask_resized)
        coordenadas_baches = self._filtrar_regiones(labeled_mask)
        #self._dibujar_regiones_filtradas(labeled_mask, coordenadas_baches) #Opcional,(Debuging) para ver mascaras de segmentacion
        return coordenadas_baches
