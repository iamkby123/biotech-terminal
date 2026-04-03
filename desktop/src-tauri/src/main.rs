// Biotech Terminal Desktop — Tauri wrapper
// Launches FastAPI server as a child process, opens native window

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::process::{Command, Child};
use std::sync::Mutex;
use std::path::PathBuf;

static SERVER_PROCESS: Mutex<Option<Child>> = Mutex::new(None);

fn find_app_dir() -> PathBuf {
    // Strategy 1: Look for api.py relative to the exe location
    if let Ok(exe_path) = std::env::current_exe() {
        // Try various parent levels (works for both dev and release)
        let mut dir = exe_path.clone();
        for _ in 0..8 {
            if let Some(parent) = dir.parent() {
                dir = parent.to_path_buf();
                if dir.join("api.py").exists() {
                    return dir;
                }
            } else {
                break;
            }
        }
    }

    // Strategy 2: Look relative to current working directory
    if let Ok(cwd) = std::env::current_dir() {
        if cwd.join("api.py").exists() {
            return cwd;
        }
        // Try parent
        if let Some(parent) = cwd.parent() {
            if parent.join("api.py").exists() {
                return parent.to_path_buf();
            }
        }
    }

    // Strategy 3: Hardcoded fallback for development
    let dev_path = PathBuf::from(r"C:\Users\kbysn\Desktop\training1\website\biotech-terminal");
    if dev_path.join("api.py").exists() {
        return dev_path;
    }

    // Last resort
    PathBuf::from(".")
}

fn start_fastapi_server() -> Result<(), String> {
    let app_dir = find_app_dir();

    if !app_dir.join("api.py").exists() {
        return Err(format!(
            "Could not find api.py in: {}\n\nMake sure you run the app from the biotech-terminal directory,\nor place the exe next to api.py.",
            app_dir.display()
        ));
    }

    println!("[Biotech Terminal] Starting FastAPI server in: {:?}", app_dir);

    match Command::new("python")
        .args(&["-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", "8000"])
        .current_dir(&app_dir)
        .spawn()
    {
        Ok(child) => {
            println!("[Biotech Terminal] FastAPI server started (PID: {})", child.id());
            *SERVER_PROCESS.lock().unwrap() = Some(child);
            Ok(())
        }
        Err(e) => {
            Err(format!(
                "Failed to start Python server: {}\n\nMake sure Python is installed and in your PATH.\nApp directory: {}",
                e, app_dir.display()
            ))
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
    // Start FastAPI server
    if let Err(err) = start_fastapi_server() {
        // Show error in a message box on Windows
        #[cfg(target_os = "windows")]
        {
            use std::ptr;
            use std::ffi::CString;
            extern "system" {
                fn MessageBoxA(hwnd: *mut u8, text: *const i8, caption: *const i8, utype: u32) -> i32;
            }
            let text = CString::new(err.clone()).unwrap_or_default();
            let caption = CString::new("Biotech Terminal - Startup Error").unwrap_or_default();
            unsafe { MessageBoxA(ptr::null_mut(), text.as_ptr(), caption.as_ptr(), 0x10); }
        }
        eprintln!("[ERROR] {}", err);
        return;
    }

    // Run Tauri app
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .on_window_event(|_window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                stop_server();
            }
        })
        .run(tauri::generate_context!())
        .expect("Failed to run Biotech Terminal");

    stop_server();
}
