#!/usr/bin/env bash
# Launch the Tauri CLI with the host GTK stack.
# VS Code's snap exports GSETTINGS_SCHEMA_DIR and GTK_PATH into the terminal.
# Host WebKit then aborts: the snap schemas do not match system GTK.
set -euo pipefail
cd "$(dirname "$0")/.."
unset GSETTINGS_SCHEMA_DIR GTK_PATH GTK_EXE_PREFIX GTK_IM_MODULE_FILE \
  GDK_PIXBUF_MODULEDIR GDK_PIXBUF_MODULE_FILE GIO_MODULE_DIR GTK_MODULES
if [ -n "${XDG_DATA_DIRS_VSCODE_SNAP_ORIG:-}" ]; then
  export XDG_DATA_DIRS="$XDG_DATA_DIRS_VSCODE_SNAP_ORIG"
fi
exec npx tauri "$@"
