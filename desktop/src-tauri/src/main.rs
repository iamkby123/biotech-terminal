// Biotech Terminal Desktop — Tauri wrapper
// Launches FastAPI server as a child process, opens native window

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::process::{Command, Child};
use std::sync::Mutex;

static SERVER_PROCESS: Mutex<Option<Child>> = Mutex::new(None);

fn start_fastapi_server() {
    // Navigate up from desktop/src-tauri to the app root
    let app_dir = std::env::current_exe()
        .ok()
        .and_then(|p| {
            // In dev: desktop/src-tauri/target/debug/biotech-terminal.exe → go up 4 levels to desktop/, then up 1 to app root
            // In production: we'll be in the app directory
            p.parent()
                .and_then(|p| p.parent())
                .and_then(|p| p.parent())
                .and_then(|p| p.parent())
                .and_then(|p| p.parent())
                .map(|p| p.to_path_buf())
        })
        .unwrap_or_else(|| {
            // Fallback: try current directory's parent
            std::env::current_dir()
                .ok()
                .and_then(|p| p.parent().map(|pp| pp.to_path_buf()))
                .unwrap_or_else(|| std::path::PathBuf::from(".."))
        });

    println!("[Biotech Terminal] Starting FastAPI server in: {:?}", app_dir);

    match Command::new("python")
        .args(&["-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", "8000"])
        .current_dir(&app_dir)
        .spawn()
    {
        Ok(child) => {
            println!("[Biotech Terminal] FastAPI server started (PID: {})", child.id());
            *SERVER_PROCESS.lock().unwrap() = Some(child);
        }
        Err(e) => {
            eprintln!("[Biotech Terminal] Failed to start server: {}", e);
        }
    }
}

fn stop_server() {
    if let Ok(mut guard) = SERVER_PROCESS.lock() {
        if let Some(ref mut child) = *guard {
            println!("[Biotech Terminal] Stopping server (PID: {})", child.id());
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn main() {
    // Start FastAPI server as background process
    start_fastapi_server();

    // Run Tauri app — the index.html loading page polls until server is ready
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .on_window_event(|_window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                stop_server();
            }
        })
        .run(tauri::generate_context!())
        .expect("Failed to run Biotech Terminal");

    // Final cleanup
    stop_server();
}
