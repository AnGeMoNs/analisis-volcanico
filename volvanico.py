import numpy as np
import matplotlib.pyplot as plt

# ====================================================================
# CONFIGURACIÓN GLOBAL Y CONSTANTES
# ====================================================================
UMBRAL_FRIO = 80
UMBRAL_MEDIO = 120
UMBRAL_CRITICO = 150
TEMP_MIN = 0
TEMP_MAX = 200
DIMENSION = 6
ARCHIVO_GRAFICO = 'estadisticas_volcan.png'

# ====================================================================
# EJERCICIO 2: ANÁLISIS TÉRMICO VOLCÁNICO
# ====================================================================

def generar_matriz_termica():
    """Genera una matriz con valores aleatorios entre TEMP_MIN y TEMP_MAX°C"""
    return np.random.randint(TEMP_MIN, TEMP_MAX + 1, size=(DIMENSION, DIMENSION))


def calcular_promedio_diagonal(matriz):
    """Calcula el promedio de la diagonal principal"""
    diagonal = np.diag(matriz)
    promedio = np.mean(diagonal)
    return promedio, diagonal


def contar_zonas_criticas(matriz):
    """Cuenta celdas con temperatura > UMBRAL_CRITICO en la mitad inferior"""
    mitad_inferior = matriz[DIMENSION//2:, :]
    zonas_criticas = np.sum(mitad_inferior > UMBRAL_CRITICO)
    total_celdas = mitad_inferior.size
    return zonas_criticas, total_celdas


def contar_rango_medio(matriz):
    """Cuenta valores entre UMBRAL_FRIO y UMBRAL_MEDIO°C"""
    en_rango = np.sum((matriz >= UMBRAL_FRIO) & (matriz <= UMBRAL_MEDIO))
    total_celdas = matriz.size
    return en_rango, total_celdas


def analizar_columna(matriz, columna):
    """Analiza si una columna tiene temperaturas > UMBRAL_MEDIO°C"""
    if not (0 <= columna < DIMENSION):
        raise ValueError(f"Columna debe estar entre 0 y {DIMENSION-1}")
    
    valores_columna = matriz[:, columna]
    tiene_alta_temp = np.any(valores_columna > UMBRAL_MEDIO)
    max_temp = np.max(valores_columna)
    
    return tiene_alta_temp, max_temp, valores_columna


def encontrar_minimo_mitad_superior(matriz):
    """Encuentra la temperatura mínima en la mitad superior"""
    mitad_superior = matriz[:DIMENSION//2, :]
    minimo = np.min(mitad_superior)
    posicion = np.unravel_index(np.argmin(mitad_superior), mitad_superior.shape)
    return minimo, posicion


def crear_graficos_volcan(matriz, nombre_archivo=ARCHIVO_GRAFICO):
    """Crea múltiples gráficos estadísticos del análisis volcánico"""
    fig = plt.figure(figsize=(16, 12))
    temps_flat = matriz.flatten()
    
    # 1. Mapa de calor principal
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(matriz, cmap='hot', interpolation='nearest', 
                     vmin=TEMP_MIN, vmax=TEMP_MAX)
    ax1.set_title('Mapa Térmico del Cráter Volcánico', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Columna', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Fila', fontsize=10, fontweight='bold')
    
    for i in range(DIMENSION):
        for j in range(DIMENSION):
            color = 'white' if matriz[i, j] > 100 else 'black'
            ax1.text(j, i, f'{matriz[i, j]}°', ha="center", va="center", 
                     color=color, fontsize=8, fontweight='bold')
    
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Temperatura (°C)', fontsize=9, fontweight='bold')
    
    # 2. Histograma de distribución
    ax2 = plt.subplot(2, 3, 2)
    n, bins, patches = ax2.hist(temps_flat, bins=20, color='orangered', 
                                edgecolor='black', alpha=0.7)
    
    for i, patch in enumerate(patches):
        if bins[i] < UMBRAL_FRIO:
            patch.set_facecolor('green')
        elif bins[i] <= UMBRAL_MEDIO:
            patch.set_facecolor('yellow')
        elif bins[i] <= UMBRAL_CRITICO:
            patch.set_facecolor('orange')
        else:
            patch.set_facecolor('red')
    
    ax2.axvline(UMBRAL_FRIO, color='blue', linestyle='--', linewidth=2, 
                label=f'Límite inferior ({UMBRAL_FRIO}°C)')
    ax2.axvline(UMBRAL_MEDIO, color='orange', linestyle='--', linewidth=2, 
                label=f'Límite medio ({UMBRAL_MEDIO}°C)')
    ax2.axvline(UMBRAL_CRITICO, color='red', linestyle='--', linewidth=2, 
                label=f'Riesgo crítico ({UMBRAL_CRITICO}°C)')
    ax2.set_xlabel('Temperatura (°C)', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Frecuencia', fontsize=10, fontweight='bold')
    ax2.set_title('Distribución de Temperaturas', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=7)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 3. Boxplot por filas
    ax3 = plt.subplot(2, 3, 3)
    datos_filas = [matriz[i, :] for i in range(DIMENSION)]
    bp = ax3.boxplot(datos_filas, labels=[f'Fila {i}' for i in range(DIMENSION)],
                     patch_artist=True, notch=True)
    
    colores_box = plt.cm.YlOrRd(np.linspace(0.3, 0.9, DIMENSION))
    for patch, color in zip(bp['boxes'], colores_box):
        patch.set_facecolor(color)
    
    ax3.axhline(UMBRAL_CRITICO, color='red', linestyle='--', linewidth=2, 
                alpha=0.7, label='Riesgo crítico')
    ax3.axhline(UMBRAL_MEDIO, color='orange', linestyle='--', linewidth=2, 
                alpha=0.7, label='Riesgo medio')
    ax3.set_xlabel('Fila del Cráter', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Temperatura (°C)', fontsize=10, fontweight='bold')
    ax3.set_title('Distribución de Temperaturas por Fila', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=7, loc='best')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    plt.xticks(rotation=45)
    
    # 4. Promedios por columna
    ax4 = plt.subplot(2, 3, 4)
    promedios_col = np.mean(matriz, axis=0)
    colores_col = ['red' if p > UMBRAL_MEDIO else 'orange' if p > UMBRAL_FRIO else 'green' 
                   for p in promedios_col]
    bars = ax4.bar(range(DIMENSION), promedios_col, color=colores_col, 
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    ax4.axhline(UMBRAL_MEDIO, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax4.set_xlabel('Columna', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Temperatura Promedio (°C)', fontsize=10, fontweight='bold')
    ax4.set_title('Temperatura Promedio por Columna', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}°', 
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 5. Mapa de contorno
    ax5 = plt.subplot(2, 3, 5)
    x = np.arange(DIMENSION)
    y = np.arange(DIMENSION)
    X, Y = np.meshgrid(x, y)
    
    contour = ax5.contourf(X, Y, matriz, levels=15, cmap='hot')
    ax5.contour(X, Y, matriz, levels=15, colors='black', linewidths=0.5, alpha=0.3)
    ax5.set_xlabel('Columna', fontsize=10, fontweight='bold')
    ax5.set_ylabel('Fila', fontsize=10, fontweight='bold')
    ax5.set_title('Mapa de Contorno Térmico', fontsize=12, fontweight='bold')
    cbar5 = plt.colorbar(contour, ax=ax5)
    cbar5.set_label('Temperatura (°C)', fontsize=9, fontweight='bold')
    
    # 6. Gráfico circular de clasificación
    ax6 = plt.subplot(2, 3, 6)
    zona_fria = np.sum(temps_flat < UMBRAL_FRIO)
    zona_media = np.sum((temps_flat >= UMBRAL_FRIO) & (temps_flat <= UMBRAL_MEDIO))
    zona_caliente = np.sum((temps_flat > UMBRAL_MEDIO) & (temps_flat <= UMBRAL_CRITICO))
    zona_critica = np.sum(temps_flat > UMBRAL_CRITICO)
    
    sizes = [zona_fria, zona_media, zona_caliente, zona_critica]
    labels = [f'Fría (<{UMBRAL_FRIO}°C)\n{zona_fria} celdas',
              f'Media ({UMBRAL_FRIO}-{UMBRAL_MEDIO}°C)\n{zona_media} celdas',
              f'Caliente ({UMBRAL_MEDIO}-{UMBRAL_CRITICO}°C)\n{zona_caliente} celdas',
              f'Crítica (>{UMBRAL_CRITICO}°C)\n{zona_critica} celdas']
    colors = ['green', 'yellow', 'orange', 'red']
    explode = [0, 0, 0.1, 0.15]
    
    ax6.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
            explode=explode, startangle=90, textprops={'fontsize': 8, 'fontweight': 'bold'})
    ax6.set_title('Clasificación de Zonas Térmicas', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
    print(f"\n✓ Gráficos guardados en '{nombre_archivo}'")
    print("✓ Mostrando gráficos en pantalla... (Cierra la ventana para continuar)")
    plt.show()
    plt.close()


def mostrar_matriz(matriz):
    """Muestra la matriz de forma ordenada"""
    print("\n" + "=" * 50)
    print("MAPA TÉRMICO DEL CRÁTER (Temperaturas en °C)")
    print("=" * 50)
    print("\n      ", end="")
    for j in range(DIMENSION):
        print(f"Col{j:>2}  ", end="")
    print()
    print("    " + "-" * (DIMENSION * 7 + 6))
    for i in range(DIMENSION):
        print(f"Fila {i} |", end="")
        for j in range(DIMENSION):
            print(f"{matriz[i][j]:>5} ", end="")
        print("|")
    print("    " + "-" * (DIMENSION * 7 + 6))


def ejecutar_opcion(matriz, opcion):
    """Ejecuta la opción seleccionada y maneja la salida"""
    if opcion == '1':
        promedio, diagonal = calcular_promedio_diagonal(matriz)
        print(f"\n=== PROMEDIO DE LA DIAGONAL PRINCIPAL ===")
        print(f"Valores de la diagonal: {diagonal}")
        print(f"Promedio: {promedio:.2f}°C")
        
    elif opcion == '2':
        zonas_criticas, total = contar_zonas_criticas(matriz)
        print(f"\n=== ZONAS DE RIESGO CRÍTICO (MITAD INFERIOR) ===")
        print(f"Celdas con temperatura > {UMBRAL_CRITICO}°C: {zonas_criticas}")
        print(f"Total de celdas en mitad inferior: {total}")
        print(f"Porcentaje de riesgo: {(zonas_criticas/total)*100:.2f}%")
        
    elif opcion == '3':
        en_rango, total = contar_rango_medio(matriz)
        print(f"\n=== CONTEO DE TEMPERATURAS EN RANGO MEDIO [{UMBRAL_FRIO}-{UMBRAL_MEDIO}]°C ===")
        print(f"Celdas en rango medio: {en_rango}")
        print(f"Total de celdas: {total}")
        print(f"Porcentaje: {(en_rango/total)*100:.2f}%")
        
    elif opcion == '4':
        try:
            columna = int(input(f"Ingrese el número de columna (0-{DIMENSION-1}): "))
            tiene_alta, max_temp, valores = analizar_columna(matriz, columna)
            print(f"\nColumna {columna}:")
            print(f"Temperaturas: {valores}")
            print(f"Temperatura máxima: {max_temp}°C")
            print(f"{'✓ Sí' if tiene_alta else '✗ No'}, la columna {columna} tiene temperaturas superiores a {UMBRAL_MEDIO}°C.")
        except (ValueError, IndexError) as e:
            print(f"Error: {e}")
            
    elif opcion == '5':
        minimo, posicion = encontrar_minimo_mitad_superior(matriz)
        print(f"\n=== TEMPERATURA MÍNIMA EN MITAD SUPERIOR ===")
        print(f"Temperatura mínima: {minimo}°C")
        print(f"Posición: Fila {posicion[0]}, Columna {posicion[1]}")
        
    elif opcion == '6':
        crear_graficos_volcan(matriz)
        
    elif opcion == '7':
        return generar_matriz_termica()
    
    return matriz


def menu_principal():
    """Función principal con menú interactivo"""
    print("=" * 60)
    print("  ANÁLISIS TÉRMICO VOLCÁNICO - INSTITUTO GEOTÉRMICO ANDINO")
    print("=" * 60)
    
    matriz = generar_matriz_termica()
    print(f"\n✓ Mapa térmico generado exitosamente ({DIMENSION}x{DIMENSION})")
    mostrar_matriz(matriz)
    
    while True:
        print("\n" + "=" * 60)
        print("MENÚ DE OPCIONES")
        print("=" * 60)
        print("1. Promedio de la diagonal principal")
        print(f"2. Zonas de riesgo crítico en la mitad inferior (>{UMBRAL_CRITICO}°C)")
        print(f"3. Conteo de temperaturas entre {UMBRAL_FRIO}°C y {UMBRAL_MEDIO}°C (rango medio)")
        print(f"4. Análisis de columna (¿hay temperaturas >{UMBRAL_MEDIO}°C?)")
        print("5. Temperatura mínima en la mitad superior")
        print("6. Generar gráficos estadísticos completos")
        print("7. Generar nueva matriz térmica")
        print("8. Salir")
        print("=" * 60)
        
        opcion = input("\nSeleccione una opción (1-8): ").strip()
        
        if opcion == '1':
            matriz = ejecutar_opcion(matriz, opcion)
        elif opcion == '2':
            matriz = ejecutar_opcion(matriz, opcion)
        elif opcion == '3':
            matriz = ejecutar_opcion(matriz, opcion)
        elif opcion == '4':
            matriz = ejecutar_opcion(matriz, opcion)
        elif opcion == '5':
            matriz = ejecutar_opcion(matriz, opcion)
        elif opcion == '6':
            crear_graficos_volcan(matriz)
        elif opcion == '7':
            matriz = generar_matriz_termica()
            print("\n✓ Nueva matriz térmica generada")
            mostrar_matriz(matriz)
        elif opcion == '8':
            print("\n¡Gracias por usar el sistema de análisis volcánico!")
            break
        else:
            print("\nOpción inválida. Intente nuevamente.")


if __name__ == "__main__":
    menu_principal()
