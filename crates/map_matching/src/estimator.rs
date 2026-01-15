/*
 * @Author: ShirahaYuki  shirhayuki2002@gmail.com
 * @Date: 2026-01-15 13:12:59
 * @LastEditors: ShirahaYuki  shirhayuki2002@gmail.com
 * @LastEditTime: 2026-01-15 15:57:45
 * @FilePath: /map_matching/src/estimator.rs
 * @Description:位置评估算法，负责初始化定位和评估位置可靠性
 *
 * Copyright (c) 2026 by ShirahaYuki, All Rights Reserved.
 */
use crate::matcher::types::SearchItem;

pub struct Estimator {
    his_pos: Vec<SearchItem>,
}

impl Estimator {
    pub fn new() -> Self {
        Self {
            his_pos: Vec::new(),
        }
    }

    pub fn estimate(new_point: SearchItem) -> Option<SearchItem> {
        todo!()
    }
}
