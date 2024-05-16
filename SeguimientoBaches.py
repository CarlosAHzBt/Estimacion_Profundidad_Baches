import numpy as np
import logging

# Configuración básica de logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class SeguimientoBaches:
    def __init__(self, tolerancia_posicion_base=200, tolerancia_tamano=50, tolerancia_temporal=3):
        self.historial_baches = []
        self.tolerancia_posicion_base = tolerancia_posicion_base  # Aumentado a 200 píxeles
        self.tolerancia_tamano = tolerancia_tamano  # Aumentado a 50 píxeles
        self.tolerancia_temporal = tolerancia_temporal  # Tolerancia temporal en frames
        self.baches_procesados = set()

    def agregar_bache(self, bache):
        self.historial_baches.append(bache)

    def agrupar_baches(self):
        agrupaciones = []
        self.baches_procesados = set()  # Reiniciar la lista de baches procesados antes de cada agrupación
        baches_por_frame = self.organizar_baches_por_frame()

        for frame_actual, baches_actuales in baches_por_frame.items():
            if frame_actual == 0:
                continue

            baches_previos = baches_por_frame.get(frame_actual - 1, [])

            matriz_costura = self.crear_matriz_costura(baches_previos, baches_actuales)

            for bache_actual, bache_previo in matriz_costura.items():
                if bache_previo and self.es_mismo_bache(bache_actual, bache_previo):
                    grupo = self.encontrar_grupo(agrupaciones, bache_previo)
                    if grupo:
                        grupo.append(bache_actual)
                        self.log_bache_agrupado(bache_actual, grupo)
                    else:
                        agrupaciones.append([bache_previo, bache_actual])
                    self.baches_procesados.update(bache_actual.frames_agrupados)
                else:
                    grupo = [bache_actual]
                    agrupaciones.append(grupo)

        return agrupaciones

    def organizar_baches_por_frame(self):
        baches_por_frame = {}
        for bache in self.historial_baches:
            frame_num = self.extraer_numero_frame(bache.id_bache)
            if frame_num not in baches_por_frame:
                baches_por_frame[frame_num] = []
            baches_por_frame[frame_num].append(bache)
        
        # Ordenar los baches en cada frame por id_bache
        for frame_num in baches_por_frame:
            baches_por_frame[frame_num].sort(key=lambda b: b.id_bache)
        
        return dict(sorted(baches_por_frame.items()))

    def crear_matriz_costura(self, baches_previos, baches_actuales):
        matriz_costura = {}
        for bache_actual in baches_actuales:
            mejor_bache_previo = None
            menor_distancia = float('inf')
            for bache_previo in baches_previos:
                distancia = np.linalg.norm(np.array(bache_actual.centro_circulo) - np.array(bache_previo.centro_circulo))
                logging.debug(f'Comparando {bache_actual.id_bache} con {bache_previo.id_bache}: Distancia={distancia}')
                if distancia < menor_distancia and distancia < self.tolerancia_posicion_base:
                    menor_distancia = distancia
                    mejor_bache_previo = bache_previo
            matriz_costura[bache_actual] = mejor_bache_previo
            logging.debug(f'Asociación: {bache_actual.id_bache} -> {mejor_bache_previo.id_bache if mejor_bache_previo else "None"} (Distancia: {menor_distancia})')
        return matriz_costura

    def encontrar_grupo(self, agrupaciones, bache):
        for grupo in agrupaciones:
            if bache in grupo:
                return grupo
        return None

    def es_mismo_bache(self, bache1, bache2):
        if bache2 is None:
            return False
        distancia = np.linalg.norm(np.array(bache1.centro_circulo) - np.array(bache2.centro_circulo))
        diferencia_radio = abs(bache1.radio_circulo_bache_px - bache2.radio_circulo_bache_px)
        diferencia_temporal = abs(self.extraer_numero_frame(bache1.id_bache) - self.extraer_numero_frame(bache2.id_bache))

        logging.debug(f'Comparando: {bache1.id_bache} con {bache2.id_bache} (Distancia: {distancia}, Diferencia Radio: {diferencia_radio}, Diferencia Temporal: {diferencia_temporal})')
        
        return (
            distancia < self.tolerancia_posicion_base and
            diferencia_radio < self.tolerancia_tamano and
            diferencia_temporal <= self.tolerancia_temporal
        )

    def extraer_numero_frame(self, id_bache):
        try:
            return int(id_bache.split('_')[1])
        except (IndexError, ValueError):
            return 0

    def calcular_promedios(self, agrupaciones):
        promedios = []
        for grupo in agrupaciones:
            if len(grupo) > 0:
                promedio_radio = np.mean([b.radio_maximo for b in grupo])
                promedio_profundidad = np.mean([b.profundidad_del_bache_estimada for b in grupo])
                bache_promedio = grupo[0]
                bache_promedio.radio_maximo = promedio_radio
                bache_promedio.profundidad_del_bache_estimada = promedio_profundidad
                bache_promedio.frames_agrupados = [b.id_bache for b in grupo]
                promedios.append(bache_promedio)
        return promedios

    def log_bache_agrupado(self, bache, grupo):
        ids_baches_existentes = [b.id_bache for b in grupo]
        logging.info(f'Bache {bache.id_bache} agrupado con los baches existentes: {", ".join(ids_baches_existentes)}')
