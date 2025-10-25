import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords


def leer_documentos(ruta):
    with open(ruta, 'r') as f:
        documentos = f.read()
    return documentos

def cargar_documento(categoria):
    textos = ""
    for i in range(1, 7):
        # inserte aquí la ruta hasta la base de conocimiento
        ruta_base = "/Users/davidjaque/Dev/Programación científica/Laboratorio 1/Base de conocimiento/"
        ruta_categoria = ruta_base + categoria
        ruta_texto = ruta_categoria + "/Texto " + str(i)
        textos += leer_documentos(ruta_texto) + "\n"
    return textos

def datos():
    datos = ""
    datos += cargar_documento("Ciencia")
    datos += cargar_documento("Cultura")
    datos += cargar_documento("Deportes")
    datos += cargar_documento("Tecnología")
    return datos

def normalizacion(palabra):
    # pasar a minúsculas
    palabra = palabra.lower()
    # eliminar saltos de línea y tabulaciones
    palabra = palabra.replace("\n", " ").replace("\t", " ")
    # eliminar signos y caracteres especiales comunes
    caracteres = "!¡#¿?$%&*:;.,()[]{}\"'/-_—…<>@^+=`~"
    for c in caracteres:
        palabra = palabra.replace(c, "")
    # eliminar tildes y diacríticos
    palabra = ''.join(
        ch for ch in unicodedata.normalize('NFD', palabra)
        if unicodedata.category(ch) != 'Mn'
    )
    
    return palabra

def datos_normalizados(datos):
    datos_normalizados = ""
    for p in datos:
        datos_normalizados += normalizacion(p)
    return datos_normalizados

def tokenizar(datos_norm):
    tokens = datos_norm.split(" ")
    return tokens

def del_stopwords(tokens):
    # Recibe una lista de tokens y elimina las stopwords, devuelve la lista sin las stopwords
    # se cargan las stopwords  en un conjunto (set) para búsquedas rápidas
    stopwords_es = set(stopwords.words('spanish'))
    
    # Filtrar los tokens usando una comprensión de listas:
    tokens_filtrados = [
        token for token in tokens if token not in stopwords_es
    ]
    return tokens_filtrados

def main():
    
    print(datos())
    print("\n\n***  NORMALIZAR  ***\n\n")
    d = datos()
    dn= datos_normalizados(d)
    print(dn)
    print("\n\n***  TOKENIZAR  ***\n\n")

    tokens = tokenizar(dn)
    print(tokens)

    print("\n\n***  SIN STOPWORDS  ***\n\n")
    print(del_stopwords(tokens))

main()