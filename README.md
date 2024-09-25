# Descripción General del Proyecto

Este proyecto está diseñado para detectar y analizar baches en carreteras utilizando datos capturados por cámaras Intel RealSense almacenados en archivos `.bag`. El sistema procesa estos archivos para extraer imágenes RGB y de profundidad, aplica un modelo de segmentación para identificar baches, estima sus dimensiones y profundidades utilizando datos de nube de puntos, y genera un informe detallado en formato CSV.

## Herramientas y Librerías Utilizadas

- **Python 3.x**: Lenguaje de programación principal del proyecto.
- **PyRealSense2**: Librería para interactuar con cámaras Intel RealSense y procesar archivos `.bag`.
- **OpenCV (cv2)**: Utilizada para procesamiento y manipulación de imágenes.
- **NumPy**: Para operaciones numéricas y manejo de arreglos.
- **Open3D**: Librería para procesamiento de datos 3D y nubes de puntos.
- **PyTorch**: Framework de aprendizaje profundo utilizado para cargar y ejecutar el modelo de segmentación.
- **scikit-image**: Para procesamiento de imágenes, como redimensionamiento y etiquetado de regiones.
- **Shapely**: Para operaciones geométricas, como manejar puntos y polígonos.
- **Pillow (PIL)**: Para operaciones adicionales con imágenes.
- **Tkinter**: Interfaz gráfica de usuario para seleccionar carpetas de entrada y salida.
- **Logging**: Para registrar información y errores durante la ejecución.
- **Concurrent Futures**: Para procesamiento concurrente y mejorar el rendimiento.

## Estructura del Proyecto y Funcionalidades

### 1. Extracción de Información de Archivos .bag

- **ProcesadorBags y BagFile**: Estas clases se encargan de procesar los archivos `.bag`. Extraen imágenes RGB y de profundidad, y opcionalmente nubes de puntos en formato `.ply`.

#### Funcionamiento:
- Configuran una tubería (pipeline) para leer los datos de los archivos `.bag`.
- Extraen y guardan cada cuadro (frame) como imágenes en carpetas organizadas.

### 2. Carga y Aplicación del Modelo de Segmentación

- **CargarModelo y ModeloSegmentacion**: Se encargan de cargar el modelo de segmentación entrenado y aplicarlo a las imágenes extraídas.

#### Funcionamiento:
- Cargan un modelo de segmentación (por ejemplo, un Segformer entrenado con PyTorch).
- Procesan las imágenes para identificar áreas que corresponden a baches.
- Generan máscaras de segmentación para localizar los baches en las imágenes.

### 3. Procesamiento de Baches Detectados

- **Bache**: Clase que representa un bache detectado y contiene métodos para calcular sus propiedades.

#### Funcionamiento:
- Calcula el contorno del bache a partir de las coordenadas de la máscara de segmentación.
- Estima el tamaño (radio y diámetro) y la profundidad del bache utilizando datos de nube de puntos.
- Genera imágenes con los contornos y anotaciones superpuestas.

### 4. Administración de Archivos

- **AdministradorDeArchivos**: Maneja las operaciones de archivos y directorios, como crear carpetas, guardar imágenes y limpiar datos temporales.

#### Funcionamiento:
- Genera listas de archivos e imágenes para procesar.
- Crea las estructuras de directorios necesarias para guardar resultados.
- Elimina archivos temporales si es necesario.

### 5. Seguimiento y Agrupación de Baches

- **SeguimientoBaches**: Rastrea los baches detectados a través de múltiples frames para agrupar detecciones que corresponden al mismo bache físico.

#### Funcionamiento:
- Usa criterios espaciales y temporales para determinar si los baches en diferentes frames son el mismo.
- Agrupa baches y calcula promedios de sus dimensiones y profundidades.
- Mantiene un historial para mejorar la precisión del seguimiento.

### 6. Interfaz Gráfica de Usuario

- **Funciones `select_folder` y `main`**: Proporcionan una interfaz gráfica simple para que el usuario seleccione las carpetas de entrada y salida.

#### Funcionamiento:
- Utilizan Tkinter para abrir cuadros de diálogo de selección de carpetas.
- Inician el proceso principal una vez que se han seleccionado las rutas.

### 7. Generación del Informe

- **Método `generar_documento_de_deterioros` en la clase `Main`**:
  - Compila los datos procesados de los baches y genera un archivo CSV llamado `deterioros.csv`.
  - El informe incluye información como ID del bache, radio máximo promedio, profundidad promedio, ruta de la imagen y los frames agrupados.

## Cómo Funciona el Proyecto (Paso a Paso)

### 1. Ejecución Inicial:

Al ejecutar el script principal, se abre una interfaz para seleccionar la carpeta que contiene los archivos `.bag` y la carpeta de salida.

### 2. Extracción de Datos:

- Los archivos `.bag` seleccionados son procesados para extraer imágenes RGB y de profundidad.
- Las imágenes son guardadas en una estructura de directorios organizada.

### 3. Carga del Modelo de Segmentación:

- El modelo pre-entrenado de segmentación es cargado en memoria.
- Se verifica si hay una GPU disponible para acelerar el procesamiento.

### 4. Aplicación del Modelo:

- El modelo de segmentación es aplicado a cada imagen extraída.
- Se identifican las regiones que corresponden a baches.

### 5. Procesamiento de Baches:

Para cada bache detectado:
- Se calcula su contorno y se estima su tamaño en píxeles.
- Se convierte el tamaño de píxeles a metros utilizando la escala calculada a partir de la altura de captura.
- Se utiliza la nube de puntos para estimar la profundidad del bache.
- Se generan imágenes con los contornos y medidas anotadas.

### 6. Seguimiento y Agrupación:

- Los baches son rastreados a través de diferentes frames para identificar si corresponden al mismo bache físico.
- Se agrupan las detecciones y se calculan promedios de sus dimensiones.

### 7. Generación del Informe:

- Se crea un archivo CSV que contiene todos los baches detectados y sus propiedades.
- El informe facilita el análisis posterior y la toma de decisiones.

### 8. Limpieza:

- Opcionalmente, se pueden eliminar los archivos extraídos para ahorrar espacio en disco.

## Cómo Ejecutar el Proyecto

### Requisitos Previos

- **Python 3.8** instalado en el sistema.

### Instalación de las librerías requeridas:

```bash
pip install numpy opencv-python open3d pyrealsense2 torch torchvision scikit-image shapely pillow tkinter
```

**Nota**: Algunas librerías como `pyrealsense2` y `open3d` pueden requerir dependencias adicionales o instrucciones específicas de instalación.

## Pasos para Ejecutar

### 1. Preparación del Entorno:
- Clona o descarga el repositorio del proyecto en tu máquina local.
- Asegúrate de que todos los archivos Python están en el directorio correcto o ajusta las importaciones según sea necesario.
- Verifica que el archivo del modelo de segmentación (`model_state_dictV18-este_ya_trae_ruido.pth`) está en la ruta correcta o modifica el código para apuntar a su ubicación.

### 2. Ejecución del Script Principal:
- Ejecuta el script que contiene la función `main()`, generalmente es el archivo `main.py`.
- Usa el siguiente comando en la terminal:

```bash
python main.py
```

### 3. Selección de Carpetas:
- Se abrirán cuadros de diálogo para seleccionar:
  - La carpeta que contiene los archivos `.bag`.
  - La carpeta donde deseas que se guarden los resultados.

### 4. Procesamiento:
- El programa comenzará a procesar los archivos.
- Se mostrará información en la terminal o en el registro (`logging`) sobre el progreso.

### 5. Resultados:
- Una vez finalizado, encontrarás el archivo `deterioros.csv` en la carpeta de salida seleccionada.
- También habrá imágenes y otros datos generados durante el procesamiento.
