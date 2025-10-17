def leer_documentos(ruta):
    with open(ruta, 'r') as f:
        documentos = f.read()
    return documentos

def cargar_documento(categoria):
    textos = ""
    for i in range(1, 7):
        ruta = f"/Users/davidjaque/Dev/Programación científica/Laboratorio 1/Base de conocimiento/{categoria}/Texto {i}"
        textos += leer_documentos(ruta) + "\n"
    return textos

def datos():
    datos = ""
    datos += cargar_documento("Ciencia")
    datos += cargar_documento("Cultura")
    datos += cargar_documento("Deportes")
    datos += cargar_documento("Tecnología")
    return datos

def main():
    print(datos())

main()