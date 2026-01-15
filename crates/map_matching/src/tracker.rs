use crate::{
    estimator::Estimator,
    extractor::{extractor::Extractor, types::ExtractorCfg},
    matcher::{matcher::Matcher, types::MatcherCfg, vector_client::VectorClient},
};
use tracing::info;

pub struct MapMatcher {
    matcher: Matcher,
    extractor: Extractor,
    client: VectorClient,
    estimator: Estimator,
}

impl MapMatcher {
    #[tracing::instrument(level = "info")]
    pub fn new(macther_cfg: MatcherCfg, extractor_cfg: ExtractorCfg) -> anyhow::Result<Self> {
        info!("▶Creating Matcher and Extractor");
        let client = VectorClient::new(macther_cfg.client_addr.clone());
        let matcher = Matcher::new(macther_cfg)?;
        let extractor = Extractor::new(extractor_cfg)?;
        let estimator = Estimator::new();
        info!("✔Matcher and Extractor has been Created");
        Ok(Self {
            matcher,
            extractor,
            client,
            estimator,
        })
    }
}

//     fn init(
//         &mut self,
//         drone_img: &Mat,
//         utm_x: f32,
//         utm_y: f32,
//         crs: String,
//         radius: f32,
//         k: usize,
//     ) {
//         // 先拿图像处理得到特征向量
//         let feature_vec = self.backend.forword(drone_img).unwrap();
//         //到数据库里查询
//         let search_res = self
//             .client
//             .search(SearchRequest {
//                 vector: feature_vec,
//                 utm_x,
//                 utm_y,
//                 crs,
//                 radius,
//                 k,
//             })
//             .unwrap();
//     }
// }
