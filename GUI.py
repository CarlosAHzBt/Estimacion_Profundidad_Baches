import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

def select_folder(title):
    root = tk.Tk()
    root.withdraw()  # Ocultar la ventana raíz
    folder_path = filedialog.askdirectory(title=title)
    root.destroy()  # Cerrar la ventana raíz después de la selección
    return folder_path

def main():
    input_folder = select_folder("Seleccionar carpeta de archivos bag")
    if not input_folder:
        messagebox.showerror("Error", "No se seleccionó la carpeta de archivos bag.")
        return

    output_folder = select_folder("Seleccionar carpeta de salida")
    if not output_folder:
        messagebox.showerror("Error", "No se seleccionó la carpeta de salida.")
        return

    # Aquí puedes agregar el código para pasar estas rutas al resto de tu aplicación.
    print("Carpeta de entrada seleccionada:", input_folder)
    print("Carpeta de salida seleccionada:", output_folder)
    # Ejemplo de cómo podrías iniciar el procesamiento:
    if output_folder and input_folder :
        from Main import Main

        main_process = Main(input_folder, output_folder)
        main_process.run()

if __name__ == "__main__":
    main()
