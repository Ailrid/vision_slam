use std::time::Instant;
pub struct PredictPoint {
    // 后端类型,onnx或者vino
    pub x: f64,
    pub y: f64,
    pub z: f64,
    // 时间与帧编号
    pub time: Instant,
    pub frame_id: usize,
}

pub struct FramePriori {
    pub utm_x: f32,
    pub utm_y: f32,
    pub crs: String,
    pub radius: f32,
    pub k: usize,
}
impl Default for FramePriori {
    fn default() -> Self {
        Self {
            utm_x: 0.0,
            utm_y: 0.0,
            crs: "".to_string(),
            radius: 500.0,
            k: 2,
        }
    }
}
