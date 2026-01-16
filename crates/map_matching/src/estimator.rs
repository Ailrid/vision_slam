/*
 * @Author: ShirahaYuki  shirhayuki2002@gmail.com
 * @Date: 2026-01-15 13:12:59
 * @LastEditors: ShirahaYuki  shirhayuki2002@gmail.com
 * @LastEditTime: 2026-01-16 18:02:42
 * @FilePath: /map_matching/src/estimator.rs
 * @Description:位置评估算法，负责初始化定位和评估位置可靠性
 *
 * Copyright (c) 2026 by ShirahaYuki, All Rights Reserved.
 */
use crate::types::PredictPoint;

pub struct Estimator {
    //每次地图匹配找到的结果
    buffer_points: Vec<Vec<PredictPoint>>,
    tracking_points: Vec<PredictPoint>,
}

impl Estimator {
    pub fn new() -> Self {
        Self {
            buffer_points: Vec::new(),
            tracking_points: Vec::new(),
        }
    }
    /// 更新点坐标集合，这里面的点都是
    pub fn update(&mut self, frame_info: Vec<PredictPoint>) {
        // 添加新的结果
        self.buffer_points.push(frame_info);
    }
    ///初始化定位程序
    pub fn init(&mut self) -> bool {
        //首先，判断帧数是否达到要求
        if self.buffer_points.len() < 5 {
            return false;
        }
        todo!();
    }
    //已经初始化后，利用卡尔曼滤波进行点的跟踪
    pub fn tracking(&mut self) {
        todo!();
    }
}
