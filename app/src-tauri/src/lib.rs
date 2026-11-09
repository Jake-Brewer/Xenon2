mod code_api;
mod example_bank;
mod feedback;
mod ffi;
mod folder_agent;
mod gpu_temp;
mod inference;
mod memory;
mod model_parts;
mod persistence;
mod sloth_memory;
mod voice;

use std::path::PathBuf;
use tauri::{Manager, PhysicalPosition, PhysicalSize};

/// Open maximized on the monitor the window is actually on, so a 1920x1080 laptop
/// panel and a larger external display both get a window that fills the work area
/// (the space below the menu bar, not under it). If the saved position is off every
/// display, move onto the primary monitor first. Wayland often rejects set_position;
/// maximize() is what makes the window fit.
fn fit_window_to_screen(app: &tauri::App) {
    let Some(window) = app.get_webview_window("main") else {
        return;
    };
    let Ok(monitors) = window.available_monitors() else {
        let _ = window.maximize();
        return;
    };
    let on_a_screen = window.outer_position().ok().zip(window.outer_size().ok()).is_some_and(|(pos, size)| {
        monitors.iter().any(|m| {
            let area = m.work_area();
            let x1 = pos.x.max(area.position.x);
            let y1 = pos.y.max(area.position.y);
            let x2 = (pos.x + size.width as i32).min(area.position.x + area.size.width as i32);
            let y2 = (pos.y + size.height as i32).min(area.position.y + area.size.height as i32);
            x2 - x1 > 48 && y2 - y1 > 48
        })
    });
    if !on_a_screen {
        if let Some(monitor) = window.primary_monitor().ok().flatten().or_else(|| monitors.into_iter().next()) {
            let area = monitor.work_area();
            let _ = window.set_position(PhysicalPosition::new(area.position.x, area.position.y));
            let _ = window.set_size(PhysicalSize::new(area.size.width, area.size.height));
        }
    }
    if let Err(err) = window.maximize() {
        eprintln!("xenon2: maximize failed ({err}); sizing to the work area");
        let monitor = window
            .current_monitor()
            .ok()
            .flatten()
            .or_else(|| window.primary_monitor().ok().flatten());
        if let Some(monitor) = monitor {
            let area = monitor.work_area();
            let _ = window.set_position(PhysicalPosition::new(area.position.x, area.position.y));
            let _ = window.set_size(PhysicalSize::new(area.size.width, area.size.height));
        }
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            // Repo root = two levels up from this crate (app/src-tauri -> app -> <repo root>),
            // so the app finds Phase 1's model/vocab files without needing them copied into
            // app/ or bundled -- fine for this dev-focused phase; portable/USB-relative paths
            // are Phase 6's job (see PLAN.md).
            let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            let repo_root = manifest_dir
                .parent()
                .expect("app/ dir")
                .parent()
                .expect("repo root")
                .to_path_buf();

            fit_window_to_screen(app);

            let (engine_state, model_info, device_info) = inference::load_engine(&repo_root);
            // Local coding API for the VS Code extension (vscode-xenon/). See code_api.rs.
            code_api::start(app.handle().clone(), engine_state.clone(), model_info.0.clone());
            app.manage(engine_state);
            app.manage(model_info);
            app.manage(device_info);
            // Phase 6: export/import need to find models/, inference-engine/data/, and
            // voice-pipeline/models/ relative to the repo root too, so it's kept as managed state
            // instead of being recomputed from CARGO_MANIFEST_DIR a second time.
            app.manage(memory::RepoRoot(repo_root.clone()));

            // Phase 7: spawn the voice-pipeline sidecar (see voice.rs's module doc). Model
            // loading happens inside the sidecar process itself (a few seconds for STT/TTS), so
            // this doesn't block the app's own startup -- voice_ready() lets the frontend poll
            // for when it's actually usable, and a spawn failure (e.g. voice-pipeline's venv
            // isn't set up in this checkout) degrades to "voice unavailable" rather than
            // panicking, since typed chat must keep working regardless.
            let voice_process = voice::spawn_voice_process(&repo_root, app.handle().clone());
            app.manage(voice::VoiceState(voice_process));

            // Model load blocks this closure, and the window is mapped only after it
            // returns. Measure (and re-maximize) once the compositor has the window.
            let handle = app.handle().clone();
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(800));
                let handle_for_fit = handle.clone();
                let _ = handle.run_on_main_thread(move || {
                    if let Some(window) = handle_for_fit.get_webview_window("main") {
                        // Wayland ignores most moves. Still ask for the primary monitor
                        // (the one with the panel) before maximizing, so a second display
                        // does not keep the window on the other panel.
                        if let Some(primary) = window.primary_monitor().ok().flatten() {
                            let area = primary.work_area();
                            let _ = window.set_position(PhysicalPosition::new(
                                area.position.x,
                                area.position.y,
                            ));
                        }
                        let _ = window.maximize();
                    }
                });
                std::thread::sleep(std::time::Duration::from_millis(600));
                let handle_for_log = handle.clone();
                let _ = handle.run_on_main_thread(move || {
                    let Some(window) = handle_for_log.get_webview_window("main") else {
                        return;
                    };
                    let maxed = window.is_maximized().unwrap_or(false);
                    let size = window.outer_size().ok();
                    let pos = window.outer_position().ok();
                    let monitor = window
                        .current_monitor()
                        .ok()
                        .flatten()
                        .or_else(|| window.primary_monitor().ok().flatten());
                    let work = monitor.as_ref().map(|monitor| {
                        let area = monitor.work_area();
                        format!(
                            "{}x{} at {},{} ({})",
                            area.size.width,
                            area.size.height,
                            area.position.x,
                            area.position.y,
                            monitor.name().map(|name| name.as_str()).unwrap_or("")
                        )
                    });
                    if let (Some(size), Some(monitor)) = (size, monitor.as_ref()) {
                        let area = monitor.work_area();
                        // A dock or top bar already makes a maximized window a little
                        // smaller than the monitor's full mode. Only step in when the
                        // window is still the un-maximized startup size.
                        let too_small = size.width > 0
                            && size.height > 0
                            && (size.width + 160 < area.size.width
                                || size.height + 160 < area.size.height);
                        if !maxed && too_small {
                            let _ = window.set_position(PhysicalPosition::new(
                                area.position.x,
                                area.position.y,
                            ));
                            let _ = window
                                .set_size(PhysicalSize::new(area.size.width, area.size.height));
                        }
                    }
                    eprintln!(
                        "xenon2: window maximized={maxed} outer={size:?} pos={pos:?} work={work:?}"
                    );
                });
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            inference::generate_response,
            inference::stop_generation,
            inference::get_model_name,
            inference::get_inference_device,
            persistence::save_conversation_file,
            persistence::open_conversation_file,
            persistence::pick_save_dialog,
            persistence::pick_open_dialog,
            persistence::default_conversation_path,
            persistence::load_session_file,
            persistence::save_session_file,
            memory::load_settings,
            memory::save_settings,
            memory::effective_conversations_dir,
            memory::pick_folder_dialog,
            memory::export_memory,
            memory::import_memory,
            feedback::record_feedback,
            feedback::feedback_summary,
            sloth_memory::list_sloth_facts,
            sloth_memory::delete_sloth_fact,
            sloth_memory::clear_sloth_facts,
            voice::voice_ready,
            voice::voice_listen,
            voice::voice_listen_cancel,
            voice::voice_speak_start,
            voice::voice_speak_feed,
            voice::voice_speak_finish,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
