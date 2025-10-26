import unicodedata  # Para quitar tildes
import math        # Para el logaritmo de IDF
import os          # Para construir rutas de archivos (un poco mejor que sumar strings)

# --- 0. CONFIGURACIÓN ---

# ¡IMPORTANTE! Pon aquí la ruta a tu carpeta "Base de conocimiento"
ruta_base = "/Users/davidjaque/Dev/Programación científica/Laboratorio 1/Base de conocimiento/"

categorias = ["Ciencia", "Cultura", "Deportes", "Tecnología"]

# El valor K para la votación (puedes cambiarlo)
k_vecinos = 3

# Se descargan  las stopwords de NLTK 
try:
    from nltk.corpus import stopwords
    STOPWORDS_ES = set(stopwords.words('spanish'))
except LookupError:
    print("Descargando lista de 'stopwords' de NLTK...")
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    STOPWORDS_ES = set(stopwords.words('spanish'))


# --- 1. PREPROCESAMIENTO (CRITERIO 1) ---

def normalizar_texto(texto):
    """
    Toma un string y lo limpia:
    1. Pasa a minúsculas
    2. Quita saltos de línea y tabulaciones
    3. Quita signos de puntuación y caracteres especiales
    4. Quita tildes
    """
    texto = texto.lower()
    texto = texto.replace("\n", " ").replace("\t", " ")
    
    # Lista de caracteres a eliminar
    caracteres_malos = "!¡#¿?$%&*:;.,()[]{}\"'/-_—…<>@^+=`~"
    for char in caracteres_malos:
        texto = texto.replace(char, "")
        
    # Quitar números (a veces pueden ser ruido)
    texto = ''.join(c for c in texto if not c.isdigit())
        
    # Quitar tildes (esto las descompone en 'letra' + 'tilde' y quita la 'tilde')
    texto_normalizado = ""
    for ch in unicodedata.normalize('NFD', texto):
        if unicodedata.category(ch) != 'Mn':
            texto_normalizado += ch
            
    return texto_normalizado.strip() 

def tokenizar_y_limpiar(texto):
    """
    Toma un texto ya normalizado y hace 2 cosas:
    1. Tokeniza (lo separa en una lista de palabras)
    2. Elimina las stopwords (palabras comunes como 'el', 'la', 'de')
    """
    # 1. Tokeniza
    palabras = texto.split() # Separa por espacios -por defecto-
    
    # 2. Elimina stopwords
    palabras_limpias = []
    for p in palabras:
        if p not in STOPWORDS_ES and len(p) > 1: # Filtra stopwords y palabras de 1 letra
            palabras_limpias.append(p)
            
    return palabras_limpias


# --- 2. CÁLCULO MANUAL DE TF-IDF ---

def calcular_tf(doc_tokens):
    """
    Calcula la Frecuencia de Término (TF) para un SOLO documento.
    TF = (N° de veces que aparece una palabra) / (Total de palabras en el doc)
    Devuelve un diccionario: {'palabra': 0.05, 'otra_palabra': 0.02}
    """
    total_palabras_doc = len(doc_tokens)
    if total_palabras_doc == 0:
        return {} # Evita división por cero si el doc está vacío

    tf_valores = {}
    for palabra in doc_tokens:
        # Aquí se cuenta cuantas veces está la palabra
        if palabra not in tf_valores:
            tf_valores[palabra] = 0 # Si es la primera vewz que aparece
        tf_valores[palabra] += 1
        
    # Se dividde por el total para obtener la frecuencia
    for palabra in tf_valores:
        tf_valores[palabra] = tf_valores[palabra] / total_palabras_doc
        
    return tf_valores

def calcular_idf(corpus_procesado, vocabulario):
    """
    Calcula la Frecuencia Inversa de Documento (IDF) para CADA palabra
    en el vocabulario

    IDF = log(N / (1 + DF(t)))
    - N = Número total de documentos
    - DF(t) = Cuántos documentos contienen la palabra 't'
    Devuelve un diccionario: {'palabra': 1.5, 'otra_palabra': 2.1}
    """
    N = len(corpus_procesado) # N° total de documentos
    df_valores = {}
    idf_valores = {}
    
    # 1. Contar DF (Document Frequency)
    # Inicializamos el contador para cada palabra del vocabulario
    for palabra in vocabulario:
        df_valores[palabra] = 0
    
    # Recorremos cada documento
    for doc_tokens in corpus_procesado:
        # S eutiliza set() para contar cada palabra solo UNA vez por documento
        palabras_unicas_doc = set(doc_tokens) 
        
        for palabra in palabras_unicas_doc:
            if palabra in df_valores: # Si la palabra está en el vocabulario
                df_valores[palabra] += 1

    # 2. Calcular IDF
    for palabra in vocabulario:
        df = df_valores[palabra]
        idf_valores[palabra] = math.log(N / (1 + df)) 
        
    return idf_valores

def crear_vectores_tfidf(corpus_procesado, vocabulario, idf_valores):
    """
    Crea la matriz TF-IDF.
    Será una lista de listas (vectores), donde cada vector representa un documento.
    El valor de cada palabra en el vector es: TF * IDF
    """
    matriz_tfidf = [] # matriz (lista de vectores)
    
    # Se recorre cada documento (ya tokenizado y limpio)
    for doc_tokens in corpus_procesado:
        
        # 1. Calculamos el TF para ESTE documento
        tf_doc = calcular_tf(doc_tokens) # Ej: {'hola': 0.1, 'mundo': 0.05}
        
        # 2. Creamos el vector TF-IDF para este documento
        vector_doc = []
        
        # Recorremos el vocabulario EN ORDEN.
        # Esto es CLAVE para que todos los vectores tengan el mismo orden.
        for palabra in vocabulario:
            
            # Obtenemos el TF de la palabra (si no está, su TF es 0)
            tf = tf_doc.get(palabra, 0)
            
            # Obtenemos el IDF de la palabra (ya lo calculamos)
            idf = idf_valores[palabra]
            
            # Calculamos el peso final
            peso_tfidf = tf * idf
            vector_doc.append(peso_tfidf)
            
        # Agregamos el vector de este documento a nuestra matriz
        matriz_tfidf.append(vector_doc)
        
    return matriz_tfidf


# --- 3. SIMILITUD DEL COSENO ---

def similitud_coseno(vec1, vec2):
    """
    Calcula la similitud del coseno entre dos vectores (dos listas de números).
    Fórmula: (A · B) / (||A|| * ||B||)
    """
    producto_punto = 0
    magnitud_vec1 = 0
    magnitud_vec2 = 0
    
    # Recorremos los vectores (ambos tienen el mismo largo: el del vocabulario)
    for i in range(len(vec1)):
        # 1. Producto punto (Sumatoria de A[i] * B[i])
        producto_punto += vec1[i] * vec2[i]
        
        # 2. Magnitudes (Sumatoria de A[i]^2 y B[i]^2)
        magnitud_vec1 += vec1[i] ** 2
        magnitud_vec2 += vec2[i] ** 2
        
    # Terminamos las magnitudes sacando la raíz cuadrada
    magnitud_vec1 = math.sqrt(magnitud_vec1)
    magnitud_vec2 = math.sqrt(magnitud_vec2)
    
    # 3. División
    if magnitud_vec1 == 0 or magnitud_vec2 == 0:
        return 0 # Evita división por cero para q no se caiga
    else:
        return producto_punto / (magnitud_vec1 * magnitud_vec2)


# --- 4. CLASIFICACIÓN ---

def clasificar_query(query_texto, matriz_tfidf, etiquetas, vocabulario, idf_valores):
    """
    Toma un texto nuevo (query), lo compara con todos los documentos
    y predice su categoría.
    """
    
    # 1. Procesar la Query (igual que los documentos del corpus)
    query_norm = normalizar_texto(query_texto)
    query_tokens = tokenizar_y_limpiar(query_norm)
    
    # 2. Vectorizar la Query (Calcular su vector TF-IDF)
    # Usamos los IDF que YA calculamos con el corpus
    
    # 2.1. Calcular TF de la query
    tf_query = calcular_tf(query_tokens) # Ej: {'partido': 0.1, 'futbol': 0.1}
    
    # 2.2. Construir el vector
    vector_query = []
    for palabra in vocabulario:
        # TF (si la palabra no está en la query, es 0)
        tf = tf_query.get(palabra, 0)
        
        # IDF (si la palabra es nueva y no está en el vocabulario, su IDF es 0)
        idf = idf_valores.get(palabra, 0) 
        
        vector_query.append(tf * idf)

    # 3. Calcular Similitud con TODOS los documentos
    similitudes = []
    for i in range(len(matriz_tfidf)):
        vector_doc = matriz_tfidf[i]
        etiqueta_doc = etiquetas[i]
        
        # Comparamos la query con el documento 'i'
        sim = similitud_coseno(vector_query, vector_doc)
        
        # Guardamos el resultado (puntaje, etiqueta)
        similitudes.append((sim, etiqueta_doc))
        
    # 4. Ordenar resultados (de MAYOR a MENOR similitud)
    # Usamos una función simple para decirle a sorted() que ordene por el primer item (puntaje)
    def obtener_puntaje(item):
        return item[0] # item es (puntaje, etiqueta)
    
    similitudes_ordenadas = sorted(similitudes, key=obtener_puntaje, reverse=True)
    
    # 5. Seleccionar los K vecinos más cercanos (no sombrear la variable global k_vecinos)
    vecinos_seleccionados = similitudes_ordenadas[0:k_vecinos]  # los k mejores (lista de tuplas (sim, etiqueta))
    
    # 6. Votación Mayoritaria
    votos = {}
    for sim, etiqueta in vecinos_seleccionados:
        votos[etiqueta] = votos.get(etiqueta, 0) + 1
        
    # 7. Encontrar al ganador
    categoria_ganadora = ""
    max_votos = -1
    for etiqueta, conteo in votos.items():
        if conteo > max_votos:
            max_votos = conteo
            categoria_ganadora = etiqueta
            
    # Se devuelven los resultados para explicarlos
    return categoria_ganadora, vecinos_seleccionados, votos


# --- 5. FUNCIÓN PRINCIPAL ---

def main():
    """
    Esta es la función "orquesta" que llama a todas las demás en orden.
    """
    
    # --- PASO A: Cargar la Base de Conocimiento ---
    print("Iniciando Proceso...")
    print(f"Cargando base de conocimiento desde: {ruta_base}")
    
    corpus_crudo = []  # Lista con los 24 textos (strings)
    etiquetas = []     # Lista con las 24 etiquetas (ej: "Ciencia", "Ciencia"...)
    
    for cat in categorias:
        ruta_categoria = os.path.join(ruta_base, cat)
        print(f"  Cargando categoría: {cat}")
        
        for i in range(1, 7): # 6 documentos por categoría
            nombre_archivo = f"Texto {i}"
            ruta_archivo = os.path.join(ruta_categoria, nombre_archivo)
            
            try:
                with open(ruta_archivo, 'r', encoding='utf-8') as f:
                    texto = f.read()
                
                corpus_crudo.append(texto)
                etiquetas.append(cat)

            except FileNotFoundError:
                print(f"    ¡ERROR! No se encontró el archivo: {ruta_archivo}")
            except Exception as e:
                print(f"    ¡ERROR! No se pudo leer {ruta_archivo}: {e}")

    print(f"\nSe cargaron {len(corpus_crudo)} documentos.")
    if len(corpus_crudo) == 0:
        print("¡Error fatal! No se cargaron documentos. Revisa tu RUTA_BASE.")
        return

    # --- PASO B: Preprocesar el Corpus ---
    print("Preprocesando corpus (normalizar, tokenizar, stopwords)...")
    corpus_procesado = [] # Lista de listas de tokens (ej: [['hola', 'mundo'], ['sol', 'luna']])
    for doc in corpus_crudo:
        texto_norm = normalizar_texto(doc)
        tokens_limpios = tokenizar_y_limpiar(texto_norm)
        corpus_procesado.append(tokens_limpios)

    # --- PASO C: Construir Modelo TF-IDF ---
    print("Construyendo modelo TF-IDF...")
    
    # 1. Crear Vocabulario (todas las palabras únicas)
    vocabulario_set = set()
    for doc_tokens in corpus_procesado:
        for palabra in doc_tokens:
            vocabulario_set.add(palabra)
            
    # Se convierte a lista ordenada para que los vectores siempre tengan el mismo orden.
    vocabulario = sorted(list(vocabulario_set))
    print(f"  Vocabulario creado con {len(vocabulario)} palabras únicas.")
    
    # 2. Calcular IDF
    print("  Calculando valores IDF...")
    idf_valores = calcular_idf(corpus_procesado, vocabulario)
    
    # 3. Crear Matriz TF-IDF
    print("  Creando matriz de vectores TF-IDF...")
    matriz_tfidf = crear_vectores_tfidf(corpus_procesado, vocabulario, idf_valores)
    print("¡Modelo TF-IDF listo!")
    
    
    # --- PASO D: Probar con Queries ---
    # también están creadas en el archivo 'consultas'
    print("\n--- INICIO DE PRUEBAS DE CLASIFICACIÓN ---")
    
    # Aquí pones tus 4 textos de prueba
    queries_prueba = [
        "El telescopio James Webb estudia exoplanetas.", # Query 1 (Ciencia)
        "Lionel Messi es un atleta de fútbol.", # Query 2 (Deportes)
        "La pintura de Van Gogh y la música clásica son cultura.", # Query 3 (Cultura)
        "El nuevo iPhone de Apple tiene procesadores AMD." # Query 4 (Tecnología)
    ]
    
    for i, query in enumerate(queries_prueba):
        
        # Llamamos a nuestra función principal de clasificación
        categoria, vecinos, votos = clasificar_query(
            query, 
            matriz_tfidf, 
            etiquetas, 
            vocabulario, 
            idf_valores
        )
        
        print(f"\n--- Query {i+1}: '{query}' ---")
        print(f"  Predicción: {categoria}")
        
        # Explicación del resultado (requerido por la rúbrica)
        print(f"  Justificación (Votación de {k_vecinos} vecinos):")
        for sim, etiqueta in vecinos:
            print(f"    - Vecino '{etiqueta}' (Similitud: {sim:.4f})")
        print(f"  Votos finales: {votos}")

# --- Ejecuta el programa ---
main()