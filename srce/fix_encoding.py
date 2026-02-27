# -*- coding: utf-8 -*-
"""
fix_encoding.py
===============
Convierte archivos Python de Latin-1/CP1252 a UTF-8 de forma segura.

Ejecutar desde la raíz del proyecto (srce/):
    python fix_encoding.py

Qué hace:
    1. Detecta archivos .py con encoding NO UTF-8
    2. Los relee en Latin-1 (encoding que usó Windows/VS)
    3. Los rescribe en UTF-8 puro
    4. Añade la declaración # -*- coding: utf-8 -*- si no existe
    5. Hace backup de cada archivo antes de modificarlo (.bak)

Archivos afectados actualmente:
    - TEST_SUITE.py          (Latin-1)
    - solucionador_reimann.py (Latin-1)
"""

import os
import shutil
import sys
from pathlib import Path


def detectar_encoding(path: str) -> str:
    """Devuelve 'utf-8' o 'latin-1' según si el archivo decodifica en UTF-8."""
    with open(path, "rb") as f:
        raw = f.read()
    try:
        raw.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "latin-1"


def tiene_declaracion_utf8(lineas: list) -> bool:
    """Verifica si alguna de las primeras 2 líneas tiene coding: utf-8."""
    for linea in lineas[:2]:
        if "coding" in linea and "utf-8" in linea:
            return True
    return False


def convertir_a_utf8(path: str, dry_run: bool = False) -> bool:
    """
    Convierte un archivo a UTF-8.
    Devuelve True si se modificó, False si ya estaba bien.
    """
    enc = detectar_encoding(path)

    if enc == "utf-8":
        return False  # ya está bien

    # Leer en encoding original
    with open(path, encoding="latin-1") as f:
        contenido = f.read()

    lineas = contenido.splitlines(keepends=True)

    # Añadir declaración UTF-8 si no existe
    if not tiene_declaracion_utf8(lineas):
        declaracion = "# -*- coding: utf-8 -*-\n"
        # Preservar shebang si existe
        if lineas and lineas[0].startswith("#!"):
            lineas.insert(1, declaracion)
        else:
            lineas.insert(0, declaracion)
        contenido = "".join(lineas)

    if dry_run:
        print(f"  [dry-run] convertiría: {path}")
        return True

    # Backup
    backup = path + ".bak"
    shutil.copy2(path, backup)
    print(f"  Backup:   {backup}")

    # Escribir UTF-8
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(contenido)

    print(f"  ✓ Convertido: {path}")
    return True


def main(directorio: str = ".", dry_run: bool = False, limpiar_bak: bool = False):
    root = Path(directorio)
    convertidos = []
    omitidos    = []

    # Extensiones a procesar
    extensiones = {".py", ".md", ".txt"}

    # Carpetas a ignorar
    ignorar = {"__pycache__", ".git", ".vs", ".venv", "venv", "env", "node_modules"}

    print(f"\n{'='*60}")
    print(f"Fix Encoding UTF-8")
    print(f"Directorio: {root.resolve()}")
    print(f"Modo: {'DRY RUN (sin cambios)' if dry_run else 'REAL (modifica archivos)'}")
    print(f"{'='*60}\n")

    for archivo in root.rglob("*"):
        # Filtrar carpetas ignoradas
        if any(parte in ignorar for parte in archivo.parts):
            continue
        if archivo.suffix not in extensiones:
            continue
        if not archivo.is_file():
            continue
        if archivo.suffix == ".bak":
            continue

        try:
            modificado = convertir_a_utf8(str(archivo), dry_run=dry_run)
            if modificado:
                convertidos.append(str(archivo))
            else:
                omitidos.append(str(archivo))
        except Exception as e:
            print(f"  ERROR en {archivo}: {e}")

    # Limpiar backups si se pide
    if limpiar_bak and not dry_run:
        print("\nLimpiando backups .bak...")
        for bak in root.rglob("*.bak"):
            bak.unlink()
            print(f"  Eliminado: {bak}")

    print(f"\n{'─'*60}")
    print(f"Convertidos : {len(convertidos)}")
    print(f"Ya en UTF-8 : {len(omitidos)}")

    if convertidos:
        print("\nArchivos convertidos:")
        for f in convertidos:
            print(f"  • {f}")

    print(f"\n{'='*60}")
    if not dry_run and convertidos:
        print("Recuerda hacer commit de los cambios:")
        print('  git add .')
        print('  git commit -m "fix: convertir encoding a UTF-8"')
    print()


if __name__ == "__main__":
    dry_run    = "--dry-run" in sys.argv or "-n" in sys.argv
    limpiar    = "--clean-bak" in sys.argv
    directorio = "."

    # Permitir pasar directorio como argumento posicional
    args_pos = [a for a in sys.argv[1:] if not a.startswith("-")]
    if args_pos:
        directorio = args_pos[0]

    main(directorio=directorio, dry_run=dry_run, limpiar_bak=limpiar)
