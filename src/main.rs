#[cfg(not(target_arch = "wasm32"))]
fn main()
{
    env_logger::init();

    // force X11 on WSL - wsl shenanigans
    unsafe { std::env::set_var("WINIT_UNIX_BACKEND", "x11"); }
    if std::env::var("LIBGL_ALWAYS_SOFTWARE").is_err() {
        unsafe { std::env::set_var("LIBGL_ALWAYS_SOFTWARE", "1"); }
    }
    lab1::run();
}

#[cfg(target_arch = "wasm32")]
fn main() {}
