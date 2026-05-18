#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 VALIDADOR DE IMPORTS PARA DASHBOARD

Este script verifica que todos los módulos y funciones necesarios
estén disponibles ANTES de lanzar el dashboard.

Uso:
    python validate_imports.py
    
Si pasa todas las validaciones, el dashboard debería funcionar correctamente.

Autor: Jorge BC & Claude
"""

import sys
from pathlib import Path


def _configure_stdio_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconf = getattr(stream, "reconfigure", None)
        if reconf is not None:
            try:
                reconf(encoding="utf-8", errors="replace")
            except (OSError, ValueError, AttributeError):
                pass


_configure_stdio_utf8()

# Colores para output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_ok(msg):
    print(f"{Colors.GREEN}✓{Colors.END} {msg}")

def print_error(msg):
    print(f"{Colors.RED}✗{Colors.END} {msg}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠{Colors.END} {msg}")

def print_info(msg):
    print(f"{Colors.BLUE}ℹ{Colors.END} {msg}")

def validate_module(module_name, description=""):
    """Valida que un módulo esté disponible."""
    try:
        __import__(module_name)
        print_ok(f"{module_name:30s} {description}")
        return True
    except ImportError as e:
        print_error(f"{module_name:30s} NO DISPONIBLE")
        if description:
            print(f"     {Colors.YELLOW}→{Colors.END} {description}")
        return False

def validate_function(module_name, function_name):
    """Valida que una función específica exista en un módulo."""
    try:
        module = __import__(module_name, fromlist=[function_name])
        func = getattr(module, function_name)
        if callable(func):
            print_ok(f"  └─ {function_name}()")
            return True
        else:
            print_error(f"  └─ {function_name} NO ES CALLABLE")
            return False
    except AttributeError:
        print_error(f"  └─ {function_name}() NO EXISTE")
        return False
    except Exception as e:
        print_error(f"  └─ Error: {e}")
        return False

def validate_attribute(module_name, attribute_name):
    """Valida que un atributo exista en un módulo, aunque no sea callable."""
    try:
        module = __import__(module_name, fromlist=[attribute_name])
        getattr(module, attribute_name)
        print_ok(f"  └─ {attribute_name}")
        return True
    except AttributeError:
        print_error(f"  └─ {attribute_name} NO EXISTE")
        return False
    except Exception as e:
        print_error(f"  └─ Error: {e}")
        return False

def main():
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}🔍 VALIDACIÓN DE IMPORTS PARA DASHBOARD{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")
    
    all_ok = True
    
    # ========================================================================
    # SECCIÓN 1: DEPENDENCIAS CORE
    # ========================================================================
    print(f"\n{Colors.BOLD}[1] DEPENDENCIAS CORE{Colors.END}")
    print("-" * 70)
    
    core_deps = [
        ('numpy', 'Computación numérica'),
        ('scipy', 'Algoritmos científicos'),
        ('matplotlib', 'Gráficos estáticos'),
        ('numba', 'Compilación JIT'),
        ('mpmath', 'Precisión arbitraria'),
    ]
    
    for module, desc in core_deps:
        if not validate_module(module, desc):
            all_ok = False
    
    # ========================================================================
    # SECCIÓN 2: STREAMLIT Y VISUALIZACIÓN
    # ========================================================================
    print(f"\n{Colors.BOLD}[2] STREAMLIT Y VISUALIZACIÓN{Colors.END}")
    print("-" * 70)
    
    viz_deps = [
        ('streamlit', 'Framework de dashboard'),
        ('plotly', 'Gráficos interactivos'),
        ('pandas', 'Manipulación de datos'),
    ]
    
    for module, desc in viz_deps:
        if not validate_module(module, desc):
            all_ok = False
    
    # ========================================================================
    # SECCIÓN 3: MOTOR PRINCIPAL (solucionador_reimann.py)
    # ========================================================================
    print(f"\n{Colors.BOLD}[3] MOTOR PRINCIPAL{Colors.END}")
    print("-" * 70)
    
    try:
        import solucionador_reimann
        print_ok("solucionador_reimann.py        Módulo cargado")
        
        # Validar funciones específicas
        if not validate_attribute('solucionador_reimann', 'CACHE'):
            all_ok = False

        motor_functions = [
            'analizar_espaciado_puntual',
            'estudiar_espaciado_vs_N',
            'espaciado_minimo',
            'calcular_espaciados',
        ]
        
        for func_name in motor_functions:
            if not validate_function('solucionador_reimann', func_name):
                all_ok = False
    
    except ImportError as e:
        print_error(f"solucionador_reimann.py        NO DISPONIBLE")
        print(f"     {Colors.YELLOW}→{Colors.END} Error: {e}")
        print(f"     {Colors.YELLOW}→{Colors.END} Asegúrate de que el archivo existe en el mismo directorio")
        all_ok = False
    
    # ========================================================================
    # SECCIÓN 4: MÓDULO DE RIGIDEZ ESPECTRAL
    # ========================================================================
    print(f"\n{Colors.BOLD}[4] MÓDULO DE RIGIDEZ ESPECTRAL{Colors.END}")
    print("-" * 70)
    
    # 4.1 Análisis de rigidez
    try:
        from src.riemann_spectral.analysis import rigidity
        print_ok("src.riemann_spectral.analysis.rigidity")
        
        if not validate_function('src.riemann_spectral.analysis.rigidity', 'delta3_dyson_mehta'):
            all_ok = False
            print_warning("     Función delta3_dyson_mehta no encontrada")
            print_info("     El dashboard usará fallback (Δ₃ = L/15 para Poisson)")
    
    except ImportError as e:
        print_error("src.riemann_spectral.analysis.rigidity  NO DISPONIBLE")
        print(f"     {Colors.YELLOW}→{Colors.END} {e}")
        all_ok = False
    
    # 4.2 Generadores
    try:
        from src.riemann_spectral.data import generators
        print_ok("src.riemann_spectral.data.generators")
        
        gen_functions = ['generar_gue_normalizado', 'generar_poisson']
        for func in gen_functions:
            if not validate_function('src.riemann_spectral.data.generators', func):
                all_ok = False
    
    except ImportError as e:
        print_error("src.riemann_spectral.data.generators     NO DISPONIBLE")
        print(f"     {Colors.YELLOW}→{Colors.END} {e}")
        all_ok = False
    
    # 4.3 Unfolding
    try:
        from src.riemann_spectral.analysis import unfolding
        print_ok("src.riemann_spectral.analysis.unfolding")
        
        unfold_functions = ['unfolding_wigner_gue', 'unfolding_tercio_central']
        for func in unfold_functions:
            if not validate_function('src.riemann_spectral.analysis.unfolding', func):
                all_ok = False
    
    except ImportError as e:
        print_error("src.riemann_spectral.analysis.unfolding  NO DISPONIBLE")
        print(f"     {Colors.YELLOW}→{Colors.END} {e}")
        all_ok = False
    
    # ========================================================================
    # SECCIÓN 5: ARCHIVOS NECESARIOS
    # ========================================================================
    print(f"\n{Colors.BOLD}[5] ARCHIVOS REQUERIDOS{Colors.END}")
    print("-" * 70)
    
    required_files = [
        ('dashboard.py', 'Dashboard principal'),
        ('config_hardware.py', 'Configuración de hardware'),
        ('requirements.txt', 'Lista de dependencias'),
    ]
    
    for filename, desc in required_files:
        if Path(filename).exists():
            print_ok(f"{filename:30s} {desc}")
        else:
            print_error(f"{filename:30s} NO ENCONTRADO")
            all_ok = False
    
    # ========================================================================
    # RESUMEN FINAL
    # ========================================================================
    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
    
    if all_ok:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✅ VALIDACIÓN EXITOSA{Colors.END}")
        print(f"\n{Colors.GREEN}Todos los módulos y funciones están disponibles.{Colors.END}")
        print(f"{Colors.GREEN}El dashboard debería funcionar correctamente.{Colors.END}")
        print(f"\n{Colors.BLUE}Para iniciar:{Colors.END}")
        print(f"    python launch.py --auto")
        print(f"    # o")
        print(f"    streamlit run dashboard.py")
        return 0
    
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}❌ VALIDACIÓN FALLIDA{Colors.END}")
        print(f"\n{Colors.RED}Algunos módulos o funciones no están disponibles.{Colors.END}")
        print(f"\n{Colors.YELLOW}Acciones recomendadas:{Colors.END}")
        print(f"  1. Instalar dependencias: pip install -r requirements.txt")
        print(f"  2. Verificar estructura de directorios:")
        print(f"     {Colors.BLUE}./dashboard.py{Colors.END}")
        print(f"     {Colors.BLUE}./solucionador_reimann.py{Colors.END}")
        print(f"     {Colors.BLUE}./src/riemann_spectral/{Colors.END}")
        print(f"  3. Verificar nombres de funciones en el código")
        return 1

if __name__ == "__main__":
    sys.exit(main())
