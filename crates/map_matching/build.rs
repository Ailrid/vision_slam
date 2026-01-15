use std::env;
use std::fs;
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};

fn main() {
    // 1. 获取当前项目的绝对路径
    let project_dir = env::var("CARGO_MANIFEST_DIR").expect("无法获取 CARGO_MANIFEST_DIR");
    
    // 建议：使用更灵活的方式定位 .venv，考虑到可能在项目根目录也可能在 crate 目录
    let ov_libs_path = PathBuf::from(&project_dir)
        .join(".venv/lib/python3.11/site-packages/openvino/libs");

    if !ov_libs_path.exists() {
        // 如果路径不存在，尝试往上一级找（针对 Workspace 结构）
        let alt_path = PathBuf::from(&project_dir)
            .parent().unwrap()
            .join(".venv/lib/python3.11/site-packages/openvino/libs");
        
        if !alt_path.exists() {
            panic!("错误：找不到 OpenVINO 库路径。请检查 .venv 是否存在于：{:?}", ov_libs_path);
        }
    }

    // 2. 自动化处理软链接
    // openvino-finder 搜索 libopenvino_c.so，但它依赖 libopenvino.so
    // 所以这两个都必须建立不带版本号的链接
    setup_openvino_symlinks(&ov_libs_path);

    // 3. 核心配置：告诉 Cargo 搜索路径（编译时）
    let libs_dir = ov_libs_path.to_str().expect("路径包含非 UTF-8 字符");
    println!("cargo:rustc-link-search=native={}", libs_dir);
    println!("cargo:rustc-link-lib=openvino");
    println!("cargo:rustc-link-lib=openvino_c");

    // 4. 解决运行时加载问题 (rpath)
    // 这让生成的二进制文件在运行时知道去哪里找 .so
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", libs_dir);

    // 5. 【关键】满足 openvino-finder 的“怪癖”
    // 通过 rustc-env，我们可以让编译出来的程序在运行时，
    // 其内嵌的环境变量中包含 OPENVINO_INSTALL_DIR
    println!("cargo:rustc-env=OPENVINO_INSTALL_DIR={}", libs_dir);
    
    // 同时设置这个，因为有些版本的 finder 优先看 LD_LIBRARY_PATH
    println!("cargo:rustc-env=LD_LIBRARY_PATH={}", libs_dir);

    println!("cargo:rerun-if-changed=build.rs");
}

/// 自动遍历目录并为所有的 .so.版本号 文件创建不带版本号的软链接
fn setup_openvino_symlinks(dir: &Path) {
    let entries = fs::read_dir(dir).expect("无法读取库目录");

    for entry in entries.flatten() {
        let path = entry.path();
        if let Some(file_name) = path.file_name().and_then(|n| n.to_str()) {
            // 匹配 libopenvino.so.XXXX 或 libopenvino_c.so.XXXX
            if (file_name.starts_with("libopenvino.so.") || file_name.starts_with("libopenvino_c.so.")) 
                && !path.is_symlink() 
            {
                // 计算目标名称：去掉最后的版本号后缀
                // 例如 libopenvino_c.so.2541 -> libopenvino_c.so
                let dest_name = if file_name.contains("openvino_c") {
                    "libopenvino_c.so"
                } else {
                    "libopenvino.so"
                };

                let dest_path = dir.join(dest_name);

                if !dest_path.exists() {
                    println!("cargo:warning=创建软链接: {} -> {}", dest_name, file_name);
                    if let Err(e) = symlink(&path, &dest_path) {
                        println!("cargo:warning=创建链接失败: {:?}", e);
                    }
                }
            }
        }
    }
}