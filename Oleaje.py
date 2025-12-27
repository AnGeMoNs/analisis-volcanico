# -*- coding: utf-8 -*-
"""
Simulación de Oleaje para la Costa de Antofagasta-Hornitos
Basado en el Modelo Espectral de Pierson-Moskowitz

Este script modela las características del oleaje en la región de Antofagasta
utilizando el espectro de Pierson-Moskowitz, apropiado para condiciones de
mar desarrollado (fully developed sea).

Desarrollado para análisis de ingeniería costera en el Pacífico Sur chileno.
"""

import numpy as np
import matplotlib.pyplot as plt

class SimuladorOleaje:
    """
    Clase principal para simular características del oleaje usando el
    espectro de Pierson-Moskowitz. Incluye cálculo de parámetros estadísticos
    y visualización de resultados.
    """
    
    def __init__(self, velocidad_viento, profundidad=None):
        """
        Inicializa la simulación con parámetros físicos del entorno.
        
        Parámetros:
        -----------
        velocidad_viento : float
            Velocidad del viento a 10m sobre la superficie del mar [m/s]
        profundidad : float, optional
            Profundidad del agua [m]. Si es None, se asume aguas profundas.
        """
        # Constantes físicas del modelo
        self.alpha = 0.0081      # Constante de Phillips (adimensional)
        self.beta = 0.74         # Constante de forma del espectro
        self.g = 9.81            # Aceleración de gravedad [m/s²]
        self.u = velocidad_viento
        self.h = profundidad
        
        # Rango de frecuencias para análisis espectral
        self.omega_min = 0.1     # Frecuencia angular mínima [rad/s]
        self.omega_max = 3.0     # Frecuencia angular máxima [rad/s]
        self.n_puntos = 1000     # Resolución del espectro
        
    def calcular_espectro(self, omega):
        """
        Calcula la densidad espectral de energía según Pierson-Moskowitz.
        
        La fórmula describe la distribución de energía del oleaje en función
        de la frecuencia para un mar completamente desarrollado.
        
        Parámetros:
        -----------
        omega : array_like
            Frecuencias angulares [rad/s]
            
        Retorna:
        --------
        S : array_like
            Densidad espectral de energía [m²·s]
        """
        # Evitamos singularidades en omega = 0
        omega_seguro = np.where(omega == 0, 1e-10, omega)
        
        # Ecuación del espectro de Pierson-Moskowitz
        S = (self.alpha * self.g**2 / omega_seguro**5) * \
            np.exp(-self.beta * (self.g / (omega_seguro * self.u))**4)
        
        return S
    
    def calcular_numero_onda(self, omega):
        """
        Determina el número de onda k usando la relación de dispersión.
        
        Para aguas profundas: ω² = g·k
        Para aguas intermedias: ω² = g·k·tanh(k·h)
        
        Parámetros:
        -----------
        omega : float or array_like
            Frecuencia angular [rad/s]
            
        Retorna:
        --------
        k : float or array_like
            Número de onda [rad/m]
        """
        if self.h is None:
            # Caso de aguas profundas - solución analítica directa
            return omega**2 / self.g

        # Caso de aguas finitas - solución numérica (Newton-Raphson) sin SciPy
        omega_arr = np.asarray(omega, dtype=float)
        omega_seguro = np.where(omega_arr == 0, 1e-12, omega_arr)

        # Estimación inicial basada en aguas profundas
        k = omega_seguro**2 / self.g

        tol = 1e-12
        max_iter = 50

        for _ in range(max_iter):
            kh = k * self.h
            tanh_kh = np.tanh(kh)

            # f(k) = ω² - g k tanh(kh)
            f = omega_seguro**2 - self.g * k * tanh_kh

            # f'(k) = -g [tanh(kh) + kh * sech²(kh)]
            sech2_kh = 1.0 / np.cosh(kh)**2
            df = -self.g * (tanh_kh + kh * sech2_kh)

            # Evitar división por cero y mantener k positivo
            df = np.where(np.abs(df) < 1e-30, -1e-30, df)

            k_new = k - f / df
            k_new = np.where(k_new <= 0, k * 0.5, k_new)

            if np.all(np.abs(k_new - k) <= tol * (1.0 + np.abs(k_new))):
                k = k_new
                break

            k = k_new

        if np.isscalar(omega):
            return float(np.asarray(k).item())
        return k
    
    def calcular_longitud_onda(self, omega):
        """
        Convierte frecuencia angular a longitud de onda física.
        
        L = 2π/k, donde k es el número de onda.
        """
        k = self.calcular_numero_onda(omega)
        longitud = 2 * np.pi / k
        return longitud
    
    def analizar_parametros_estadisticos(self, omega, S_omega):
        """
        Extrae parámetros estadísticos clave del espectro de oleaje.
        
        Calcula:
        - Altura significativa (Hs)
        - Período de pico (Tp)
        - Frecuencia de pico (ωp)
        - Energía total (m₀)
        """
        # Integración numérica del espectro (momento de orden cero)
        domega = np.diff(omega)
        S_promedio = (S_omega[:-1] + S_omega[1:]) / 2
        m0 = np.sum(S_promedio * domega)
        
        # Altura significativa: Hs = 4·√(m₀)
        Hs = 4 * np.sqrt(m0)
        
        # Identificación del pico espectral
        idx_pico = np.argmax(S_omega)
        omega_pico = omega[idx_pico]
        T_pico = 2 * np.pi / omega_pico
        
        return {
            'altura_significativa': Hs,
            'periodo_pico': T_pico,
            'frecuencia_pico': omega_pico,
            'energia_total': m0,
            'indice_pico': idx_pico
        }
    
    def ejecutar_simulacion(self):
        """
        Ejecuta el pipeline completo de simulación y retorna resultados.
        """
        # Generación del dominio de frecuencias
        self.omega = np.linspace(self.omega_min, self.omega_max, self.n_puntos)
        
        # Cálculo del espectro de energía
        self.S_omega = self.calcular_espectro(self.omega)
        
        # Cálculo de longitudes de onda correspondientes
        self.longitudes_onda = self.calcular_longitud_onda(self.omega)
        
        # Cálculo de períodos
        self.periodos = 2 * np.pi / self.omega
        
        # Análisis estadístico
        self.resultados = self.analizar_parametros_estadisticos(
            self.omega, self.S_omega
        )
        
        return self.resultados
    
    def visualizar_resultados(self):
        """
        Genera visualizaciones profesionales de los resultados.
        """
        if not hasattr(self, 'resultados'):
            raise RuntimeError("Debe ejecutar ejecutar_simulacion() primero")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            'Análisis de Oleaje - Costa de Antofagasta (Hornitos)',
            fontsize=16, fontweight='bold'
        )
        
        # Gráfico 1: Espectro vs Frecuencia Angular
        ax = axes[0, 0]
        ax.plot(self.omega, self.S_omega, 'b-', linewidth=2, label='Espectro PM')
        ax.axvline(
            self.resultados['frecuencia_pico'],
            color='r', linestyle='--',
            label=f"ωₚ = {self.resultados['frecuencia_pico']:.3f} rad/s"
        )
        ax.set_xlabel('Frecuencia Angular ω [rad/s]', fontsize=11)
        ax.set_ylabel('Densidad Espectral S(ω) [m²·s]', fontsize=11)
        ax.set_title('Espectro de Pierson-Moskowitz', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Gráfico 2: Espectro vs Período
        ax = axes[0, 1]
        ax.plot(self.periodos, self.S_omega, 'g-', linewidth=2)
        ax.axvline(
            self.resultados['periodo_pico'],
            color='r', linestyle='--',
            label=f"Tₚ = {self.resultados['periodo_pico']:.2f} s"
        )
        ax.set_xlabel('Período T [s]', fontsize=11)
        ax.set_ylabel('Densidad Espectral S(ω) [m²·s]', fontsize=11)
        ax.set_title('Distribución de Energía por Período', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.invert_xaxis()
        
        # Gráfico 3: Longitud de Onda vs Frecuencia
        L_pico = self.calcular_longitud_onda(self.resultados['frecuencia_pico'])
        ax = axes[1, 0]
        ax.plot(self.omega, self.longitudes_onda, 'm-', linewidth=2)
        ax.axhline(
            L_pico, color='r', linestyle='--',
            label=f"Lₚ = {L_pico:.1f} m"
        )
        ax.set_xlabel('Frecuencia Angular ω [rad/s]', fontsize=11)
        ax.set_ylabel('Longitud de Onda L [m]', fontsize=11)
        ax.set_title('Relación de Dispersión', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Gráfico 4: Energía vs Longitud de Onda
        ax = axes[1, 1]
        ax.plot(self.longitudes_onda, self.S_omega, 'c-', linewidth=2)
        ax.axvline(
            L_pico, color='r', linestyle='--',
            label=f"Lₚ = {L_pico:.1f} m"
        )
        ax.set_xlabel('Longitud de Onda L [m]', fontsize=11)
        ax.set_ylabel('Densidad Espectral S(ω) [m²·s]', fontsize=11)
        ax.set_title('Energía vs Longitud de Onda', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.invert_xaxis()
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def generar_reporte(self):
        """
        Genera un reporte técnico con los resultados clave.
        """
        if not hasattr(self, 'resultados'):
            raise RuntimeError("Debe ejecutar ejecutar_simulacion() primero")
        
        L_pico = self.calcular_longitud_onda(self.resultados['frecuencia_pico'])
        
        reporte = f"""
{'='*70}
INFORME DE SIMULACIÓN DE OLEAJE
Costa de Antofagasta - Sector Hornitos
{'='*70}

Condiciones de entrada:
  • Velocidad del viento (10m): {self.u:.2f} m/s
  • Constante de Phillips (α): {self.alpha}
  • Constante de forma (β): {self.beta}
  • Profundidad: {'Profunda (> L/2)' if self.h is None else f'{self.h:.1f} m'}

Parámetros de salida del oleaje:
  • Altura significativa (Hs): {self.resultados['altura_significativa']:.2f} m
  • Período de pico (Tp): {self.resultados['periodo_pico']:.2f} s
  • Frecuencia de pico: {self.resultados['frecuencia_pico']:.3f} rad/s
  • Longitud de onda de pico: {L_pico:.1f} m
  • Energía total (momento cero): {self.resultados['energia_total']:.3f} m²

Rango de longitudes de onda calculadas:
  • Mínima: {self.longitudes_onda[-1]:.1f} m
  • Máxima: {self.longitudes_onda[0]:.1f} m
{'='*70}
"""
        return reporte


# ==================== EJECUCIÓN PRINCIPAL ====================

if __name__ == "__main__":
    # Configuración para condiciones típicas de Antofagasta
    velocidad_viento_tipica = 12.0  # m/s (ajustado para el Pacífico Sur)
    
    # Crear instancia del simulador
    simulador = SimuladorOleaje(velocidad_viento=velocidad_viento_tipica)
    
    # Ejecutar simulación completa
    print("Iniciando simulación de oleaje...")
    resultados = simulador.ejecutar_simulacion()
    
    # Generar visualizaciones
    print("Generando gráficos...")
    simulador.visualizar_resultados()
    
    # Mostrar reporte técnico
    print("\nObteniendo resultados finales...")
    reporte = simulador.generar_reporte()
    print(reporte)
    
    print("Simulación completada exitosamente.")
