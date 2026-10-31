// Cross-platform launcher for `npm run tauri`.
//
// On Linux, a VS Code snap terminal exports GSETTINGS_SCHEMA_DIR and GTK_PATH.
// Host WebKit then aborts because those schemas do not match system GTK.
// Drop them before starting Tauri. Windows has no GTK session, so this only
// forwards the existing environment and runs the local Tauri CLI.

import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const appDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const env = { ...process.env };

if (process.platform !== "win32") {
  for (const key of [
    "GSETTINGS_SCHEMA_DIR",
    "GTK_PATH",
    "GTK_EXE_PREFIX",
    "GTK_IM_MODULE_FILE",
    "GDK_PIXBUF_MODULEDIR",
    "GDK_PIXBUF_MODULE_FILE",
    "GIO_MODULE_DIR",
    "GTK_MODULES",
  ]) {
    delete env[key];
  }
  if (env.XDG_DATA_DIRS_VSCODE_SNAP_ORIG) {
    env.XDG_DATA_DIRS = env.XDG_DATA_DIRS_VSCODE_SNAP_ORIG;
  }
}

const tauriJs = path.join(appDir, "node_modules", "@tauri-apps", "cli", "tauri.js");
const child = spawn(process.execPath, [tauriJs, ...process.argv.slice(2)], {
  cwd: appDir,
  env,
  stdio: "inherit",
});

child.on("exit", (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
    return;
  }
  process.exit(code ?? 1);
});
