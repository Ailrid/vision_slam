/*
 * @Author: ShirahaYuki  shirhayuki2002@gmail.com
 * @Date: 2026-01-15 13:21:00
 * @LastEditors: ShirahaYuki  shirhayuki2002@gmail.com
 * @LastEditTime: 2026-01-16 20:32:40
 * @FilePath: /map_matching/src/location.rs
 * @Description:地图定位模块，负责给出地图定位结果
 *
 * Copyright (c) 2026 by ShirahaYuki, All Rights Reserved.
 */
use crate::{
    errors::LocationError,
    extractor::{extractor::Extractor, types::ExtractorCfg},
    matcher::{matcher::Matcher, types::MatcherCfg},
    types::{FramePriori, PredictPoint},
};
use geodesy::prelude::*;
use opencv::core::Mat;
use std::time::Instant;
use tracing::info;
pub struct Location {
    matcher: Matcher,
    extractor: Extractor,
    minimal: Minimal,
}

impl Location {
    pub fn new(
        macther_cfg: MatcherCfg,
        extractor_cfg: ExtractorCfg,
    ) -> Result<Self, LocationError> {
        info!("▶Creating Matcher and Extractor");
        let matcher = Matcher::new(macther_cfg)?;
        let extractor = Extractor::new(extractor_cfg)?;
        let minimal = Minimal::default();
        info!("✔Matcher and Extractor has been Created");
        Ok(Self {
            matcher,
            extractor,
            minimal,
        })
    }
    pub fn frame_location(
        &mut self,
        drone_img: &Mat,
        time: Instant,
        frame_id: usize,
        frame_priori: FramePriori,
    ) -> Result<Vec<PredictPoint>, LocationError> {
        //首先，用匹配模型生成特征向量并查询数据库
        let feature_vector = self.matcher.query_img(drone_img, frame_priori)?;

        let mut frame_pos = Vec::new();
        for item in &feature_vector.items {
            //查询抠图
            let sat_img = self.matcher.crop_pos(256, item)?;
            //计算每个图像的单应性矩阵
            let result = self.extractor.homography_matrix(drone_img, &sat_img);
            match result {
                Ok((_h_mat, center_point)) => {
                    // 能得到单应性矩阵结果的话，就保存这张图像的数据
                    // 计算这个中心点反演之后的位置坐标在utm下的坐标
                    let utm_x = center_point.x * item.res[0] + item.utm_x;
                    let utm_y = center_point.y * item.res[1] + item.utm_y;
                    // 转换为WGPS下的空间直角坐标系
                    let pipeline = format!("inv utm zone={} | cart", item.utm_zone);
                    let op = self.minimal.op(&pipeline)?;
                    // Coor3D 分别代表(Easting, Northing, Height)
                    let mut data = [Coor3D::raw(utm_x as f64, utm_y as f64, 0.0)];
                    // 执行转换到空间直角坐标系
                    self.minimal.apply(op, Fwd, &mut data)?;
                    let ecef_center = data[0].0;
                    // 保存这帧的位置
                    frame_pos.push(PredictPoint {
                        x: ecef_center[0],
                        y: ecef_center[1],
                        z: ecef_center[2],
                        time: time,
                        frame_id: frame_id,
                    });
                }
                Err(e) => {
                    // 计算单应性矩阵错误表明不可能匹配，直接丢掉
                    tracing::warn!("Homography matrix error:{:?}", e);
                }
            }
        }
        Ok(frame_pos)
    }
}
