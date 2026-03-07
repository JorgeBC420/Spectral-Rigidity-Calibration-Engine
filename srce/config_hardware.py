# -*- coding: utf-8 -*-
"""
⚡ Optimización de Hardware para Intel i7-1255U (12th Gen Alder Lake)

Este módulo configura automáticamente Numba, NumPy y otros backends
para aprovechar óptimamente la arquitectura híbrida P+E cores.

Arquitectura i7-1255U:
- 2 P-cores (Performance) @ 4.7 GHz: Hyper-Threading → 4 threads
- 8 E-cores (Efficiency) @ 3.5 GHz: Sin HT → 8 threads
- Total: 10 cores físicos, 12 threads lógicos
- Caché L3: 12 MB compartida
- RAM: 16 GB DDR4/DDR5

Autor: Jorge BC & Claude
Fecha: Febrero 2026
"""

import os
import platform
import psutil
import warnings
from typing import Dict, Tuple

# ============================================================================
# DETECCIÓN DE HARDWARE
# ============================================================================

def detectar_cpu() -> Dict[str, any]:
    """
    Detecta capacidades del CPU.
    
    Returns:
        Dict con información del procesador
    """
    info = {
        'procesador': platform.processor(),
        'cores_fisicos': psutil.cpu_count(logical=False),
        'cores_logicos': psutil.cpu_count(logical=True),
        'frecuencia_actual': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
        'frecuencia_max': psutil.cpu_freq().max if psutil.cpu_freq() else 0,
        'es_intel': 'Intel' in platform.processor(),
        'es_amd': 'AMD' in platform.processor(),
    }
    
    # Detectar arquitectura híbrida Intel 12th+ gen
    procesador = platform.processor().upper()
    if '12TH' in procesador or '13TH' in procesador or '14TH' in procesador:
        info['es_hibrido'] = True
        info['generacion'] = 12  # Simplificado
        
        # i7-1255U específico: 2P+8E
        if '1255U' in procesador or '1265U' in procesador:
            info['p_cores'] = 2
            info['e_cores'] = 8
        # i5-1235U: 2P+8E
        elif '1235U' in procesador:
            info['p_cores'] = 2
            info['e_cores'] = 8
        # Desktop 12700K: 8P+4E
        elif '12700' in procesador:
            info['p_cores'] = 8
            info['e_cores'] = 4
        else:
            # Heurística: mitad cores son P-cores
            info['p_cores'] = max(2, info['cores_fisicos'] // 3)
            info['e_cores'] = info['cores_fisicos'] - info['p_cores']
    else:
        info['es_hibrido'] = False
        info['p_cores'] = info['cores_fisicos']
        info['e_cores'] = 0
    
    return info


def detectar_memoria() -> Dict[str, float]:
    """Detecta capacidades de memoria."""
    mem = psutil.virtual_memory()
    
    return {
        'total_gb': mem.total / (1024**3),
        'disponible_gb': mem.available / (1024**3),
        'porcentaje_uso': mem.percent,
        'umbral_seguro_gb': (mem.available * 0.7) / (1024**3)  # Usar max 70% disponible
    }


# ============================================================================
# CONFIGURACIÓN ÓPTIMA DE THREADS
# ============================================================================

def calcular_threads_optimos(
    info_cpu: Dict,
    tipo_workload: str = "mixto"
) -> Tuple[int, str]:
    """
    Calcula número óptimo de threads según workload.
    
    Args:
        info_cpu: Información del CPU (de detectar_cpu())
        tipo_workload: 
            - "cpu_bound": Cálculo intensivo (ej: integración)
            - "memory_bound": Limitado por memoria (ej: grandes arrays)
            - "mixto": Balance general
            - "interactivo": Deja recursos para UI
    
    Returns:
        (num_threads, justificacion)
    """
    cores_fisicos = info_cpu['cores_fisicos']
    cores_logicos = info_cpu['cores_logicos']
    es_hibrido = info_cpu.get('es_hibrido', False)
    p_cores = info_cpu.get('p_cores', cores_fisicos)
    e_cores = info_cpu.get('e_cores', 0)
    
    if tipo_workload == "cpu_bound":
        if es_hibrido:
            # Para workloads intensivos: usar P-cores + algunos E-cores
            # P-cores son ~2.5x más potentes que E-cores
            threads = (p_cores * 2) + min(4, e_cores // 2)  # HT en P-cores + algunos E
            justificacion = (
                f"CPU-bound híbrido: {p_cores}P-cores (×2 HT) + {min(4, e_cores//2)} E-cores. "
                f"Prioriza P-cores de alta frecuencia."
            )
        else:
            # CPU tradicional: usar todos los cores físicos
            threads = cores_fisicos
            justificacion = f"CPU-bound tradicional: {cores_fisicos} cores físicos."
    
    elif tipo_workload == "memory_bound":
        if es_hibrido:
            # Memory-bound: menos threads para evitar saturar caché
            threads = p_cores + min(2, e_cores // 4)
            justificacion = (
                f"Memory-bound: {threads} threads para evitar saturación caché L3 "
                f"({info_cpu.get('cache_l3', '12MB')})."
            )
        else:
            threads = max(cores_fisicos // 2, 2)
            justificacion = f"Memory-bound: {threads} threads (mitad de cores)."
    
    elif tipo_workload == "interactivo":
        if es_hibrido:
            # Dejar P-cores libres para UI, usar E-cores para trabajo
            threads = min(6, e_cores)
            justificacion = (
                f"Interactivo: {threads} threads (solo E-cores). "
                f"P-cores reservados para UI."
            )
        else:
            threads = max(cores_fisicos - 2, 2)
            justificacion = f"Interactivo: {threads} threads (deja 2 cores para UI)."
    
    else:  # "mixto" (default)
        if es_hibrido:
            # Balance: P-cores con HT + mitad de E-cores
            threads = (p_cores * 2) + (e_cores // 2)
            justificacion = (
                f"Mixto híbrido: {p_cores}P (×2) + {e_cores//2}E = {threads} threads. "
                f"Balance performance/eficiencia."
            )
        else:
            # Tradicional: usar 75% de threads lógicos
            threads = max(int(cores_logicos * 0.75), 2)
            justificacion = f"Mixto: {threads} threads (75% de {cores_logicos})."
    
    # Límite superior: nunca más threads que cores lógicos
    threads = min(threads, cores_logicos)
    
    return threads, justificacion


# ============================================================================
# CONFIGURACIÓN DE NUMBA
# ============================================================================

def configurar_numba(num_threads: int = None, verbose: bool = True):
    """
    Configura Numba para máximo rendimiento.
    
    Args:
        num_threads: Número de threads (None = automático)
        verbose: Mostrar configuración aplicada
    """
    try:
        import numba
        
        # Detectar hardware si no se especificó threads
        if num_threads is None:
            info_cpu = detectar_cpu()
            num_threads, justif = calcular_threads_optimos(info_cpu, "mixto")
            if verbose:
                print(f"[Numba] Auto-detectado: {justif}")
        
        # Configurar threads
        numba.set_num_threads(num_threads)
        
        # Variables de entorno para optimización
        os.environ['NUMBA_NUM_THREADS'] = str(num_threads)
        os.environ['NUMBA_THREADING_LAYER'] = 'tbb'  # Intel TBB (mejor para Intel CPUs)
        
        # Deshabilitar warnings molestos
        numba.config.DISABLE_JIT = False
        
        if verbose:
            print(f"[Numba] Configurado: {num_threads} threads")
            print(f"[Numba] Backend: {os.environ.get('NUMBA_THREADING_LAYER', 'default')}")
        
        return True
    
    except ImportError:
        if verbose:
            warnings.warn("Numba no disponible. Instalar: pip install numba")
        return False


# ============================================================================
# CONFIGURACIÓN DE NUMPY/SCIPY
# ============================================================================

def configurar_numpy(num_threads: int = None, verbose: bool = True):
    """
    Configura NumPy/OpenBLAS/MKL para paralelización óptima.
    
    Args:
        num_threads: Número de threads (None = automático)
        verbose: Mostrar configuración aplicada
    """
    # Detectar hardware si no se especificó threads
    if num_threads is None:
        info_cpu = detectar_cpu()
        num_threads, _ = calcular_threads_optimos(info_cpu, "memory_bound")
    
    # Configurar BLAS/LAPACK threads
    # (OpenBLAS, MKL, BLIS respetan estas variables)
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(num_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(num_threads)
    
    # Para Intel MKL: afinidad de threads
    if platform.processor().startswith('Intel'):
        os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'
    
    if verbose:
        print(f"[NumPy/BLAS] Configurado: {num_threads} threads")
    
    return True


# ============================================================================
# CONFIGURACIÓN DE MATPLOTLIB
# ============================================================================

def configurar_matplotlib(backend: str = 'Agg', dpi: int = 100):
    """
    Configura Matplotlib para rendering eficiente.
    
    Args:
        backend: 'Agg' (sin GUI, rápido) o 'Qt5Agg' (con GUI)
        dpi: Resolución de figuras (100 = estándar, 150 = alta)
    """
    try:
        import matplotlib
        matplotlib.use(backend, force=True)
        matplotlib.rcParams['figure.dpi'] = dpi
        matplotlib.rcParams['savefig.dpi'] = dpi
        
        # Optimizaciones de rendering
        matplotlib.rcParams['path.simplify'] = True
        matplotlib.rcParams['path.simplify_threshold'] = 1.0
        matplotlib.rcParams['agg.path.chunksize'] = 10000
        
        print(f"[Matplotlib] Backend: {backend}, DPI: {dpi}")
        return True
    
    except ImportError:
        warnings.warn("Matplotlib no disponible")
        return False


# ============================================================================
# CONFIGURACIÓN COMPLETA AUTOMÁTICA
# ============================================================================

def configurar_sistema_automatico(
    workload: str = "mixto",
    verbose: bool = True,
    force_threads: int = None
) -> Dict[str, any]:
    """
    Configura automáticamente todo el sistema para máximo rendimiento.
    
    Args:
        workload: Tipo de carga ("cpu_bound", "memory_bound", "mixto", "interactivo")
        verbose: Mostrar información de configuración
        force_threads: Forzar número específico de threads (ignora detección)
    
    Returns:
        Dict con resumen de configuración aplicada
    """
    if verbose:
        print("="*70)
        print("⚡ CONFIGURACIÓN AUTOMÁTICA DE HARDWARE")
        print("="*70)
    
    # Detectar hardware
    info_cpu = detectar_cpu()
    info_mem = detectar_memoria()
    
    if verbose:
        print(f"\n[CPU] {info_cpu['procesador']}")
        print(f"[CPU] Cores: {info_cpu['cores_fisicos']} físicos, {info_cpu['cores_logicos']} lógicos")
        
        if info_cpu.get('es_hibrido', False):
            print(f"[CPU] Arquitectura: HÍBRIDA (Alder Lake/Raptor Lake)")
            print(f"[CPU]   P-cores: {info_cpu['p_cores']} (Performance)")
            print(f"[CPU]   E-cores: {info_cpu['e_cores']} (Efficiency)")
        else:
            print(f"[CPU] Arquitectura: TRADICIONAL")
        
        print(f"[RAM] {info_mem['total_gb']:.1f} GB total, "
              f"{info_mem['disponible_gb']:.1f} GB disponible "
              f"({info_mem['porcentaje_uso']:.1f}% uso)")
    
    # Calcular threads óptimos
    if force_threads is not None:
        num_threads = force_threads
        justificacion = f"Forzado manualmente a {force_threads} threads"
        if verbose:
            print(f"\n[Config] {justificacion}")
    else:
        num_threads, justificacion = calcular_threads_optimos(info_cpu, workload)
        if verbose:
            print(f"\n[Config] {justificacion}")
    
    # Aplicar configuraciones
    resumen = {
        'cpu': info_cpu,
        'memoria': info_mem,
        'threads_configurados': num_threads,
        'workload': workload,
        'justificacion': justificacion
    }
    
    if verbose:
        print(f"\n[Aplicando configuraciones...]")
    
    # Numba
    resumen['numba_ok'] = configurar_numba(num_threads, verbose)
    
    # NumPy/BLAS
    resumen['numpy_ok'] = configurar_numpy(num_threads, verbose)
    
    # Matplotlib (para Streamlit, usar Agg)
    resumen['matplotlib_ok'] = configurar_matplotlib('Agg', dpi=100)
    
    if verbose:
        print("\n" + "="*70)
        print("✅ CONFIGURACIÓN COMPLETADA")
        print("="*70)
        print(f"\n💡 Recomendación: Para tu i7-1255U en workload '{workload}'")
        print(f"   usar {num_threads} threads es óptimo.")
        print(f"\n⚠️  Si experimentas throttling térmico, reduce a {num_threads-2} threads.")
        print("="*70 + "\n")
    
    return resumen


# ============================================================================
# PERFILADO DE RENDIMIENTO
# ============================================================================

def benchmark_configuracion(duracion_segundos: int = 5) -> Dict[str, float]:
    """
    Ejecuta benchmark rápido para verificar configuración.
    
    Args:
        duracion_segundos: Duración del test
    
    Returns:
        Métricas de rendimiento
    """
    import time
    import numpy as np
    
    print(f"[Benchmark] Ejecutando test de {duracion_segundos}s...")
    
    # Test 1: CPU-bound (Numba)
    try:
        from numba import jit, prange
        
        @jit(nopython=True, parallel=True, fastmath=True)
        def test_cpu_bound(n):
            suma = 0.0
            for i in prange(n):
                for j in range(1000):
                    suma += np.sqrt(i * j + 1)
            return suma
        
        # Warmup
        _ = test_cpu_bound(100)
        
        # Benchmark
        inicio = time.time()
        iteraciones = 0
        while time.time() - inicio < duracion_segundos:
            _ = test_cpu_bound(10000)
            iteraciones += 1
        tiempo_cpu = time.time() - inicio
        
        ops_cpu = (iteraciones * 10000 * 1000) / tiempo_cpu / 1e6  # Mops/s
        print(f"  CPU-bound: {ops_cpu:.2f} MOps/s ({iteraciones} iters)")
    
    except Exception as e:
        ops_cpu = 0
        print(f"  CPU-bound: Error ({e})")
    
    # Test 2: Memory-bound (NumPy)
    try:
        inicio = time.time()
        iteraciones = 0
        while time.time() - inicio < duracion_segundos:
            arr = np.random.randn(10000, 1000)
            _ = np.linalg.svd(arr, full_matrices=False)
            iteraciones += 1
        tiempo_mem = time.time() - inicio
        
        mb_s = (iteraciones * 10000 * 1000 * 8) / tiempo_mem / 1e6  # MB/s
        print(f"  Memory-bound: {mb_s:.2f} MB/s ({iteraciones} iters)")
    
    except Exception as e:
        mb_s = 0
        print(f"  Memory-bound: Error ({e})")
    
    print("[Benchmark] Completado\n")
    
    return {
        'cpu_ops_ms': ops_cpu,
        'memory_mb_s': mb_s,
        'timestamp': time.time()
    }


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    # Configurar sistema automáticamente
    config = configurar_sistema_automatico(
        workload="mixto",
        verbose=True
    )
    
    # Ejecutar benchmark
    print("\n[Opcional] Benchmark de rendimiento:")
    respuesta = input("¿Ejecutar benchmark? (s/n): ")
    
    if respuesta.lower() == 's':
        metricas = benchmark_configuracion(duracion_segundos=5)
        
        print("\n📊 Resumen de Rendimiento:")
        print(f"  CPU: {metricas['cpu_ops_ms']:.2f} MOps/s")
        print(f"  Memoria: {metricas['memory_mb_s']:.2f} MB/s")
        print("\n✅ Sistema listo para computación científica\n")
