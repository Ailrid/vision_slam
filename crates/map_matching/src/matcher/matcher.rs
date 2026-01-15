/*
 * @Author: ShirahaYuki  shirhayuki2002@gmail.com
 * @Date: 2026-01-15 15:03:13
 * @LastEditors: ShirahaYuki  shirhayuki2002@gmail.com
 * @LastEditTime: 2026-01-15 15:55:38
 * @FilePath: /map_matching/src/matcher/matcher.rs
 * @Description: 地图匹配模块，负责查询地图数据库并返回匹配结果还有抠图
 *
 * Copyright (c) 2026 by ShirahaYuki, All Rights Reserved.
 */
use crate::{
    matcher::traits::MatcherBackend,
    matcher::{
        backend::{onnx_backend::OnnxBackend, vino_backend::OpenVinoBackend},
        types::{CropRequest, MatcherCfg, SearchItem, SearchRequest, SearchResponse},
        vector_client::VectorClient,
    },
};

use opencv::prelude::*;
use tracing::info;

pub struct Matcher {
    backend: Box<dyn MatcherBackend>,
    client: VectorClient,
}

impl Matcher {
    #[tracing::instrument(level = "info", fields(cfg = %cfg.backend_type))]
    pub fn new(cfg: MatcherCfg) -> anyhow::Result<Self> {
        info!("▶Creating Matcher");
        let backend: Box<dyn MatcherBackend> = match cfg.backend_type.as_str() {
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
        let client = VectorClient::new(cfg.client_addr);
        info!("✔Creating Matcher has been completed");
        Ok(Self { backend, client })
    }

    pub fn query_img(
        &mut self,
        drone_img: &Mat,
        utm_x: f32,
        utm_y: f32,
        crs: String,
        radius: f32,
        k: usize,
    ) -> anyhow::Result<SearchResponse> {
        // 先拿图像处理得到特征向量
        let feature_vec = self.backend.forword(drone_img)?;
        //到数据库里查询
        let search_res = self.client.search(SearchRequest {
            vector: feature_vec,
            utm_x,
            utm_y,
            crs,
            radius,
            k,
        })?;
        Ok(search_res)
    }
    pub fn crop_pos(&mut self, patch_size: usize, payload: SearchItem) -> anyhow::Result<Mat> {
        let crop_res = self.client.crop(CropRequest {
            patch_size,
            payload,
        })?;
        Ok(crop_res)
    }
}
