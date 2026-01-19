use nalgebra::Vector3;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct PredictPoint {
    // 坐标
    pub x: f64,
    pub y: f64,
    pub z: f64,
    // 时间与帧编号
    pub time: Instant,
    pub frame_id: usize,
    // 内点数
    pub inlier_count: usize,
    // 匹配得分
    pub score: f32,
}

impl PredictPoint {
    pub fn distance(&self, other: &PredictPoint) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt() as f32
    }
}

pub struct ENUPoint {
    pub pos: Vector3<f32>,
    pub row_pos: PredictPoint,
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
            k: 1,
        }
    }
}

pub struct ExternalParameters {
    pub velocity: Vector3<f32>,
    pub displacement: Vector3<f32>,
}
