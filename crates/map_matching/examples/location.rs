use map_matching::estimator::Estimator;
use map_matching::extractor::types::ExtractorCfg;
use map_matching::location::Location; 
use map_matching::matcher::types::MatcherCfg;
use map_matching::types::FramePriori;
use opencv::core::MatTraitConst;
use opencv::imgcodecs::{IMREAD_COLOR, imread};
use serde::Serialize;
use std::fs::{self, File};
use std::io::Write;
use std::time::Instant;

#[derive(Serialize)]
struct FinalResult {
    frame_id: usize,
    x: f64,
    y: f64,
    z: f64,
}

pub fn run_location_batch_test(
    img_dir: &str,
    output_json: &str,
    m_cfg: MatcherCfg,
    e_cfg: ExtractorCfg,
) -> anyhow::Result<()> {
    // 1. 初始化定位模块
    let mut locator = Location::new(m_cfg, e_cfg).map_err(|e| anyhow::anyhow!(e))?;

    // 2. 获取并排序所有图片路径
    let mut entries: Vec<_> = fs::read_dir(img_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "png"))
        .collect();

    // 按照文件名里的数字进行排序 (例如 "1.png", "2.png"...)
    entries.sort_by_key(|e| {
        e.file_name()
            .to_string_lossy()
            .trim_end_matches(".png")
            .parse::<usize>()
            .unwrap_or(0)
    });

    let mut results = Vec::new();

    let mut estimator = Estimator::new(5.0,5.0);
    // 3. 循环处理
    for entry in entries {
        let path = entry.path();
        let frame_id = path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .trim_end_matches(".png")
            .parse::<usize>()?;

        println!("正在处理帧: {}", frame_id);

        // 读取图片
        let img = imread(path.to_str().unwrap(), IMREAD_COLOR)?;
        if img.empty() {
            continue;
        }

        // 这里需要一个先验信息，假设你有一套默认先验，或者根据 frame_id 获取
        // 如果没有先验，这里需要构造一个空的或默认的
        let priori = FramePriori::default();

        // 执行定位
        let start_time = Instant::now();
        match locator.frame_location(&img, start_time, frame_id, priori) {
            Ok(predict_points) => {
                estimator.update(predict_points, None);
                if let Some(estimated_point) = estimator.estimate() {

                    results.push(FinalResult {
                        frame_id,
                        x: estimated_point.0,
                        y: estimated_point.1,
                        z: estimated_point.2,
                    });

                } else {
                    println!("帧 {:?} 无法估计结果,不确定当前位置", frame_id);
                }
            }
            Err(e) => {
                eprintln!("帧 {} 定位失败: {:?}", frame_id, e);
            }
        }
    }

    // 4. 保存为 JSON
    let json_data = serde_json::to_string_pretty(&results)?;
    let mut file = File::create(output_json)?;
    file.write_all(json_data.as_bytes())?;

    println!("✔ 所有结果已保存至: {}", output_json);
    Ok(())
}

fn main() {
    //初始化tracing
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_target(true) // 是否显示代码模块路径
        .init();

    let img_dir = "/home/shiraha_yuki/文档/deeplearning/database/track_png";
    let output_json = "output.json";
    let m_cfg = MatcherCfg {
        model_path: "assets/twinnet_inference.onnx".to_string(),
        backend_type: "vino".to_string(),
        ..Default::default()
    };
    let e_cfg = ExtractorCfg {
        model_path: "assets/superpoint_lightglue_pipeline.onnx".to_string(),
        backend_type: "vino".to_string(),
        ..Default::default()
    };

    run_location_batch_test(img_dir, output_json, m_cfg, e_cfg).unwrap();
}
