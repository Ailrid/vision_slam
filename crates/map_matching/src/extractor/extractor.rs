/*
 * @Author: ShirahaYuki  shirhayuki2002@gmail.com
 * @Date: 2026-01-15 13:40:49
 * @LastEditors: ShirahaYuki  shirhayuki2002@gmail.com
 * @LastEditTime: 2026-01-15 15:56:06
 * @FilePath: /map_matching/src/extractor/extractor.rs
 * @Description:特征提取和计算单应性模块
 *
 * Copyright (c) 2026 by ShirahaYuki, All Rights Reserved.
 */
use crate::extractor::backend::onnx_backend::OnnxBackend;
use crate::extractor::backend::vino_backend::OpenVinoBackend;
use crate::extractor::errors::HomographyMatrixError;
use crate::extractor::traits::ExtractorBackend;
use crate::extractor::types::{ExtractorCfg, FeatureData};
use opencv::calib3d;
use opencv::core::{Mat, MatTraitConst, Point2f, Vector, count_non_zero};

use tracing::info;
pub struct Extractor {
    backend: Box<dyn ExtractorBackend<Output = FeatureData>>,
    score_threshold: f32,
    min_inliers: usize,
    check_det: bool,
}

impl Extractor {
    #[tracing::instrument(level = "info", fields(cfg = %cfg.backend_type))]
    pub fn new(cfg: ExtractorCfg) -> anyhow::Result<Self> {
        info!("▶Creating MapExtractor");
        let backend: Box<dyn ExtractorBackend<Output = FeatureData>> =
            match cfg.backend_type.as_str() {
                "onnx" => {
                    let onnx_backend = OnnxBackend::new(
                        ort::session::builder::GraphOptimizationLevel::Level3,
                        cfg.threads,
                        &cfg.model_path,
                    )?;
                    Box::new(onnx_backend)
                }
                "vino" => {
                    let vino_backend =
                        OpenVinoBackend::new(&cfg.model_path, "", cfg.device.as_str().into())?;
                    Box::new(vino_backend)
                }
                _ => {
                    return Err(anyhow::anyhow!(
                        "Only support onnx and vino,reviced: {:?}",
                        cfg.backend_type
                    ));
                }
            };
        info!("✔Creating MapExtractor has been completed");
        Ok(Self {
            backend,
            score_threshold: cfg.score_threshold,
            min_inliers: cfg.min_inliers,
            check_det: cfg.check_det,
        })
    }

    pub fn homography_matrix(
        &mut self,
        drone_img: &Mat,
        sat_img: &Mat,
    ) -> Result<Mat, HomographyMatrixError> {
        let model_result: FeatureData = self.backend.forward(drone_img, sat_img)?;
        //无人机和地图图像的点对
        let mut src_pts = Vector::<Point2f>::new();
        let mut dst_pts = Vector::<Point2f>::new();
        //对每一对匹配的点对，拿到评分和对应的像素位置
        for i in 0..model_result.matches.shape.0 {
            let score = model_result.scores.data[i];
            //得分大于阈值才保存
            if score > self.score_threshold {
                let base_addr = i * 3;
                let idx0 = model_result.matches.data[base_addr + 1] as usize;
                let idx1 = model_result.matches.data[base_addr + 2] as usize;

                src_pts.push(Point2f::new(
                    model_result.first_kpts.data[idx0 * 2] as f32,
                    model_result.first_kpts.data[idx0 * 2 + 1] as f32,
                ));
                dst_pts.push(Point2f::new(
                    model_result.second_kpts.data[idx1 * 2] as f32,
                    model_result.second_kpts.data[idx1 * 2 + 1] as f32,
                ))
            }
        }
        // 数量初审
        if src_pts.len() < self.min_inliers {
            return Err(HomographyMatrixError::TooFewMatches(src_pts.len()));
        }

        // RANSAC 计算
        let mut mask = Mat::default();
        let h = calib3d::find_homography(&src_pts, &dst_pts, &mut mask, calib3d::USAC_MAGSAC, 5.0)?;

        if h.empty() {
            return Err(HomographyMatrixError::InvalidMatch);
        }

        // 内点数检查
        let inlier_count = count_non_zero(&mask)? as usize;
        if inlier_count < self.min_inliers {
            return Err(HomographyMatrixError::InvalidMatch);
        }

        // 检查行列式
        if self.check_det {
            // 将 Mat 转换为可以用 determinant 计算的状态
            let det = opencv::core::determinant(&h)
                .map_err(|_e| HomographyMatrixError::DegenerateMatrix)?;

            // 简化版逻辑：在 Rust 中可以直接调用 h.determinant() 如果类型匹配
            // 如果行列式太小或为负，说明投影关系是病态的
            if det.abs() < 1e-7 {
                return Err(HomographyMatrixError::DegenerateMatrix);
            }
        }

        Ok(h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::{core, imgcodecs, imgproc};

    #[test]
    fn test_extractor_homography_and_warp() -> Result<(), Box<dyn std::error::Error>> {
        let cfg = ExtractorCfg {
            backend_type: "vino".to_string(),
            model_path: "assets/superpoint_lightglue_pipeline.onnx".to_string(),
            threads: 4,
            device: "CPU".to_string(),
            min_inliers: 10,
            score_threshold: 0.5,
            check_det: true,
        };
        // 1. 初始化你的 Extractor (假设你已经有了构造函数)
        let mut extractor = Extractor::new(cfg)?;

        // 2. 读取测试图像
        // drone_img: 无人机当前的视场
        // sat_img: 卫星地图或者参考图
        let drone_img = imgcodecs::imread("assets/0.jpg", imgcodecs::IMREAD_COLOR)?;
        let sat_img = imgcodecs::imread("assets/5.jpg", imgcodecs::IMREAD_COLOR)?;

        if drone_img.empty() || sat_img.empty() {
            panic!("无法读取测试图像，请检查路径。");
        }

        // 3. 计算单应性矩阵 H
        // H 将 drone_img 中的点映射到 sat_img 的坐标系中
        match extractor.homography_matrix(&drone_img, &sat_img) {
            Ok(h) => {
                println!("单应性矩阵提取成功: {:?}", h);

                // 4. 执行反投影 (Warp Perspective)
                // 我们创建一个和卫星图一样大的画布，把无人机图“贴”上去
                let mut warped_img = Mat::default();
                imgproc::warp_perspective(
                    &drone_img,
                    &mut warped_img,
                    &h,
                    sat_img.size()?, // 映射到卫星图的尺寸
                    imgproc::INTER_LINEAR,
                    core::BORDER_CONSTANT,
                    core::Scalar::all(0.0),
                )?;

                // 5. 为了直观对比，我们将 warped_img 和 sat_img 进行叠加 (Alpha Blending)
                let mut blended = Mat::default();
                core::add_weighted(&sat_img, 0.5, &warped_img, 0.5, 0.0, &mut blended, -1)?;

                // 6. 保存结果
                // 在 Debian 13 Wayland 环境下，imwrite 是最稳妥的验证方式
                imgcodecs::imwrite("warp_result.png", &warped_img, &core::Vector::new())?;
                imgcodecs::imwrite("blended_check.png", &blended, &core::Vector::new())?;

                println!("测试完成，结果已保存至 warp_result.png 和 blended_check.png");
            }
            Err(e) => {
                // 如果两张图确实不相关，这里会打印失败原因，比如 TooFewMatches
                panic!("单应性矩阵计算失败: {:?}", e);
            }
        }

        Ok(())
    }
}
