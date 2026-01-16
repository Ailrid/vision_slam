// use map_matching::matcher::backend::vino_backend::OpenVinoBackend;
// use map_matching::matcher::traits::MatcherBackend; // 假设 extract 定义在此 trait
// use map_matching::matcher::types::{CropRequest, SearchRequest};
// use map_matching::matcher::vector_client::VectorClient;
// use opencv::{core, imgcodecs, imgproc, prelude::*};
// use openvino::DeviceType;

// #[cfg(test)]
// mod integration_tests {
//     use super::*;

//     #[test]
//     fn test_full_matching_pipeline() -> Result<(), Box<dyn std::error::Error>> {
//         // --- 1. 配置参数 ---
//         let model_xml = "assets/twinnet_inference.onnx";
//         let model_bin = "";
//         let input_drone_img_path = "assets/tile_512_3328.png";
//         let server_addr = "http://127.0.0.1:8000";

//         // --- 2. 初始化后端与客户端 ---
//         // 初始化 OpenVINO 提取器
//         let mut backend = OpenVinoBackend::new(model_xml, model_bin, DeviceType::CPU)
//             .map_err(|e| anyhow::anyhow!("VINO 后端加载失败: {:?}", e))?;

//         // 初始化请求客户端
//         let client = VectorClient::new(server_addr.to_string());

//         // --- 3. 读取并预处理本地图像 ---
//         let img_raw = imgcodecs::imread(input_drone_img_path, imgcodecs::IMREAD_COLOR)?;
//         if img_raw.empty() {
//             panic!("无法读取测试图像: {}", input_drone_img_path);
//         }

//         // 缩放到模型要求的 256x256
//         let mut resized = Mat::default();
//         imgproc::resize(
//             &img_raw,
//             &mut resized,
//             core::Size::new(256, 256),
//             0.0,
//             0.0,
//             imgproc::INTER_LINEAR,
//         )?;

//         // --- 4. 提取特征向量 ---
//         let vector = backend
//             .forword(&resized)
//             .map_err(|e| anyhow::anyhow!("特征提取失败: {:?}", e))?;

//         // --- 5. 发送 Search 请求检索数据库 ---
//         let search_req = SearchRequest {
//             vector,
//             utm_x: 0.0, // 默认不指定坐标过滤
//             utm_y: 0.0,
//             crs: "".to_string(),
//             radius: 500.0,
//             k: 1, // 只需要最匹配的一个
//         };

//         let search_res = client
//             .search(search_req)
//             .map_err(|e| anyhow::anyhow!("搜索请求失败: {:?}", e))?;

//         if search_res.items.is_empty() {
//             println!("未找到匹配结果，测试提前结束。");
//             return Ok(());
//         }

//         for (i, top_item) in search_res.items.into_iter().enumerate() {
//             println!(
//                 "匹配成功！得分: {}, 来源文件: {}",
//                 top_item.score, top_item.src
//             );
//             let output_save_path = format!("test_{}.png", i);

//             // --- 6. 发送 Crop 请求获取服务器切片 ---
//             let crop_req = CropRequest {
//                 patch_size: 256,
//                 payload: top_item, // 按照你的结构体定义，payload 包含了 SearchItem 本身
//             };

//             let cropped_mat = client
//                 .crop(crop_req)
//                 .map_err(|e| anyhow::anyhow!("扣图请求失败: {:?}", e))?;

//             // --- 7. 使用 OpenCV 保存结果 ---
//             let params = core::Vector::<i32>::new();
//             let write_success = imgcodecs::imwrite(&output_save_path, &cropped_mat, &params)?;

//             if write_success {
//                 println!("✅ 集成测试完成，切片已保存至: {}", output_save_path);
//             } else {
//                 panic!("❌ OpenCV 保存图像失败");
//             }
//         } // 拿到最匹配的一项数据 (Payload)

//         Ok(())
//     }
// }
