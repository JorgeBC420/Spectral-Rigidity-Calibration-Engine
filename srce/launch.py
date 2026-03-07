#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 LAUNCHER - Spectral Rigidity Calibration Engine

Script de inicio que:
1. Verifica dependencias
2. Configura hardware óptimamente
3. Lanza el dashboard de Streamlit

Uso:
    python launch.py                    # Modo interactivo
    python launch.py --auto             # Configuración automática
    python launch.py --threads 6        # Forzar threads específicos
    python launch.py --benchmark        # Ejecutar benchmark primero

Autor: Jorge BC & Claude
Fecha: Febrero 2026
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

# ============================================================================
# COLORES PARA TERMINAL
# ============================================================================

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_color(text, color=Colors.OKGREEN):
    """Imprime con color."""
    print(f"{color}{text}{Colors.ENDC}")

def print_header(text):
    """Imprime encabezado destacado."""
    print("\n" + "="*70)
    print_color(text, Colors.HEADER + Colors.BOLD)
    print("="*70 + "\n")

def print_success(text):
    print_color(f"✅ {text}", Colors.OKGREEN)

def print_warning(text):
    print_color(f"⚠️  {text}", Colors.WARNING)

def print_error(text):
    print_color(f"❌ {text}", Colors.FAIL)

def print_info(text):
    print_color(f"ℹ️  {text}", Colors.OKBLUE)

# ============================================================================
# VERIFICACIÓN DE DEPENDENCIAS
# ============================================================================

def verificar_python():
    """Verifica versión de Python."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print_error(f"Python {version.major}.{version.minor} detectado. Se requiere Python 3.9+")
        return False
    
    print_success(f"Python {version.major}.{version.minor}.{version.micro}")
    return True

def verificar_modulo(nombre, import_name=None, version_attr='__version__'):
    """Verifica si un módulo está instalado."""
    if import_name is None:
        import_name = nombre
    
    try:
        modulo = __import__(import_name)
        version = getattr(modulo, version_attr, 'desconocida')
        print_success(f"{nombre:20s} v{version}")
        return True
    except ImportError:
        print_error(f"{nombre:20s} NO INSTALADO")
        return False

def verificar_dependencias():
    """Verifica todas las dependencias necesarias."""
    print_header("🔍 VERIFICACIÓN DE DEPENDENCIAS")
    
    dependencias = [
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('matplotlib', 'matplotlib'),
        ('numba', 'numba'),
        ('mpmath', 'mpmath', 'version'),
        ('streamlit', 'streamlit'),
        ('plotly', 'plotly'),
        ('psutil', 'psutil'),
        ('pandas', 'pandas'),
    ]
    
    todas_ok = True
    for dep in dependencias:
        if len(dep) == 2:
            nombre, import_name = dep
            version_attr = '__version__'
        else:
            nombre, import_name, version_attr = dep
        
        if not verificar_modulo(nombre, import_name, version_attr):
            todas_ok = False
    
    if not todas_ok:
        print("\n" + "-"*70)
        print_warning("Faltan dependencias. Instalar con:")
        print_info("    pip install -r requirements.txt")
        print("-"*70 + "\n")
        return False
    
    print_success("\n✓ Todas las dependencias instaladas correctamente\n")
    return True

# ============================================================================
# CONFIGURACIÓN DE HARDWARE
# ============================================================================

def configurar_hardware(auto=True, threads=None, workload='mixto'):
    """Configura hardware usando config_hardware.py."""
    print_header("⚡ CONFIGURACIÓN DE HARDWARE")
    
    try:
        # Importar módulo de configuración
        from config_hardware import (
            configurar_sistema_automatico,
            detectar_cpu,
            detectar_memoria
        )
        
        if auto or threads is not None:
            # Configuración automática
            config = configurar_sistema_automatico(
                workload=workload,
                verbose=True,
                force_threads=threads
            )
            
            return config
        else:
            # Modo interactivo
            print_info("Detectando hardware...")
            info_cpu = detectar_cpu()
            info_mem = detectar_memoria()
            
            print(f"\n[CPU] {info_cpu['procesador']}")
            print(f"[CPU] {info_cpu['cores_fisicos']} cores físicos, "
                  f"{info_cpu['cores_logicos']} threads lógicos")
            print(f"[RAM] {info_mem['total_gb']:.1f} GB total\n")
            
            # Preguntar al usuario
            print("Tipo de workload:")
            print("  1. CPU-bound (cálculo intensivo)")
            print("  2. Memory-bound (grandes arrays)")
            print("  3. Mixto (balance)")
            print("  4. Interactivo (dejar recursos para UI)")
            
            opcion = input("\nSelecciona (1-4) [3]: ").strip() or "3"
            
            workload_map = {
                "1": "cpu_bound",
                "2": "memory_bound",
                "3": "mixto",
                "4": "interactivo"
            }
            
            workload = workload_map.get(opcion, "mixto")
            
            config = configurar_sistema_automatico(
                workload=workload,
                verbose=True
            )
            
            return config
    
    except ImportError:
        print_warning("Módulo config_hardware.py no encontrado")
        print_info("Usando configuración por defecto (4 threads)")
        
        os.environ['NUMBA_NUM_THREADS'] = '4'
        os.environ['OMP_NUM_THREADS'] = '4'
        
        return {'threads_configurados': 4}

# ============================================================================
# BENCHMARK OPCIONAL
# ============================================================================

def ejecutar_benchmark():
    """Ejecuta benchmark de rendimiento."""
    print_header("📊 BENCHMARK DE RENDIMIENTO")
    
    try:
        from config_hardware import benchmark_configuracion
        
        print_info("Ejecutando benchmark (5 segundos)...")
        metricas = benchmark_configuracion(duracion_segundos=5)
        
        print(f"\n[Resultados]")
        print(f"  CPU: {metricas['cpu_ops_ms']:.2f} MOps/s")
        print(f"  Memoria: {metricas['memory_mb_s']:.2f} MB/s")
        
        print_success("\n✓ Benchmark completado\n")
        return metricas
    
    except Exception as e:
        print_error(f"Error en benchmark: {e}")
        return None

# ============================================================================
# LANZAMIENTO DE STREAMLIT
# ============================================================================

def encontrar_dashboard():
    """Encuentra el archivo dashboard.py."""
    posibles_rutas = [
        Path(__file__).parent / "dashboard.py",
        Path(__file__).parent / "app.py",
        Path("dashboard.py"),
        Path("app.py"),
    ]
    
    for ruta in posibles_rutas:
        if ruta.exists():
            return ruta
    
    return None

def lanzar_streamlit(ruta_dashboard, port=8501):
    """Lanza Streamlit."""
    print_header("🚀 LANZANDO DASHBOARD")
    
    print_info(f"Iniciando Streamlit en puerto {port}...")
    print_info(f"Dashboard: {ruta_dashboard}")
    print(f"\n{'─'*70}")
    print_success(f"🌐 Abre tu navegador en: http://localhost:{port}")
    print(f"{'─'*70}\n")
    
    # Construir comando
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(ruta_dashboard),
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false"
    ]
    
    # Ejecutar
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n")
        print_info("Dashboard detenido por el usuario")
    except Exception as e:
        print_error(f"Error al lanzar Streamlit: {e}")
        print_info(f"Intenta manualmente: streamlit run {ruta_dashboard}")

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Launcher para Spectral Rigidity Calibration Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python launch.py                    # Modo interactivo
  python launch.py --auto             # Automático con detección
  python launch.py --threads 6        # Forzar 6 threads
  python launch.py --benchmark        # Ejecutar benchmark primero
  python launch.py --port 8080        # Usar puerto personalizado
        """
    )
    
    parser.add_argument(
        '--auto',
        action='store_true',
        help='Configuración automática de hardware'
    )
    
    parser.add_argument(
        '--threads',
        type=int,
        default=None,
        help='Número de threads a usar (omitir para auto-detectar)'
    )
    
    parser.add_argument(
        '--workload',
        choices=['cpu_bound', 'memory_bound', 'mixto', 'interactivo'],
        default='mixto',
        help='Tipo de carga de trabajo'
    )
    
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Ejecutar benchmark antes de lanzar'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8501,
        help='Puerto para Streamlit (default: 8501)'
    )
    
    parser.add_argument(
        '--skip-deps',
        action='store_true',
        help='Saltar verificación de dependencias'
    )
    
    args = parser.parse_args()
    
    # Banner
    print(Colors.BOLD + Colors.OKBLUE)
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║   🔬 SPECTRAL RIGIDITY CALIBRATION ENGINE                    ║
    ║                                                               ║
    ║   Motor de Análisis de Rigidez Espectral basado en RMT      ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """ + Colors.ENDC)
    
    print_info("Versión 2.0.0 - Febrero 2026")
    print_info("Autor: Jorge BC & Claude (Anthropic)\n")
    
    # 1. Verificar Python
    if not verificar_python():
        return 1
    
    # 2. Verificar dependencias
    if not args.skip_deps:
        if not verificar_dependencias():
            print_warning("\nContinuar sin todas las dependencias puede causar errores.")
            respuesta = input("¿Continuar de todos modos? (s/N): ")
            if respuesta.lower() != 's':
                return 1
    else:
        print_info("Saltando verificación de dependencias (--skip-deps)")
    
    # 3. Configurar hardware
    config = configurar_hardware(
        auto=args.auto,
        threads=args.threads,
        workload=args.workload
    )
    
    # 4. Benchmark opcional
    if args.benchmark:
        ejecutar_benchmark()
    
    # 5. Encontrar dashboard
    ruta_dashboard = encontrar_dashboard()
    
    if ruta_dashboard is None:
        print_error("No se encontró dashboard.py ni app.py")
        print_info("Asegúrate de que el archivo existe en el directorio actual")
        return 1
    
    # 6. Lanzar Streamlit
    lanzar_streamlit(ruta_dashboard, port=args.port)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
