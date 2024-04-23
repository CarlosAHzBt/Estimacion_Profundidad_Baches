import tkinter as tk
from tkinter import filedialog, messagebox

class GUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Configuración de Análisis de Deterioros")
        self.master.geometry("400x200")

        self.input_path = None
        self.output_path = None

        tk.Label(master, text="Seleccione las carpetas requeridas para el análisis").pack(pady=10)

        tk.Button(master, text="Seleccionar Carpeta de Entrada", command=self.select_input_folder).pack()
        tk.Button(master, text="Seleccionar Carpeta de Salida", command=self.select_output_folder).pack()
        self.continue_button = tk.Button(master, text="Continuar", state=tk.DISABLED, command=self.continue_process)
        self.continue_button.pack(pady=20)

    def select_input_folder(self):
        path = filedialog.askdirectory(title="Seleccione la carpeta de archivos .bag")
        if path:
            self.input_path = path
            self.check_paths()

    def select_output_folder(self):
        path = filedialog.askdirectory(title="Seleccione la carpeta de salida")
        if path:
            self.output_path = path
            self.check_paths()

    def check_paths(self):
        if self.input_path and self.output_path:
            self.continue_button.config(state=tk.NORMAL)

    def continue_process(self):
        self.master.destroy()  # Cerrar la ventana GUI
        from main_module import run_main_process
        run_main_process(self.input_path, self.output_path)

if __name__ == "__main__":
    root = tk.Tk()
    app = GUI(root)
    root.mainloop()
