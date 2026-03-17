use nalgebra::{Matrix2x3, Matrix3, Vector2, Vector3};
use opencv::calib3d;
use opencv::core::{Mat, Point2f, Vector};
use opencv::prelude::*;

/// CameraModel Trait 定义了单目 VIO 系统中相机模型的核心行为
pub trait CameraModel {
    /// 将相机坐标系下的 3D 点 P_c 投影到像素坐标 (u, v)
    fn project(&self, point_in_c: &Vector3<f64>) -> Option<Vector2<f64>>;

    /// 将单个像素坐标转换为去畸变后的归一化坐标 (x, y, 1.0)
    fn unproject(&self, pixel: &Vector<Point2f>) -> Vec<Vector3<f64>>;

    /// 计算投影函数相对于 3D 点 P_c 的雅可比矩阵，用于后端 BA 优化
    fn projection_jacobian(&self, point_in_c: &Vector3<f64>) -> Matrix2x3<f64>;

    /// 返回图像的分辨率 (width, height)
    fn dimensions(&self) -> (usize, usize);
}

/// 针孔相机模型 (RadTan 畸变)，深度集成 OpenCV
pub struct PinholeRadTan {
    /// 内参矩阵 K (nalgebra 格式)
    pub k: Matrix3<f64>,
    /// 畸变向量: [k1, k2, p1, p2]
    pub dist: [f64; 4],
    /// 图像尺寸: (width, height)
    pub resolution: (usize, usize),

    /// 预先构造的 OpenCV 格式内参，避免重复内存分配
    cv_k: Mat,
    /// 预先构造的 OpenCV 格式畸变系数
    cv_dist: Mat,
}

impl PinholeRadTan {
    /// 构造一个新的 PinholeRadTan 相机实例
    /// fx, fy, cx, cy: 针孔模型内参
    /// k1, k2, p1, p2: 径向与切向畸变参数
    pub fn new(
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        k1: f64,
        k2: f64,
        p1: f64,
        p2: f64,
        width: usize,
        height: usize,
    ) -> Self {
        // nalgebra 格式
        let k = Matrix3::new(fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);
        // 预生产 OpenCV Mat，这在 Debian 13 环境下通过 FFI 传递时开销极低
        let cv_k = Mat::from_slice_2d(&[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
            .expect("Connot create cv_k mat")
            .to_owned();
        let cv_dist = Mat::from_slice(&[k1, k2, p1, p2])
            .expect("Connot create cv_dist mat")
            .try_clone()
            .expect("Connot create cv_dist mat");

        Self {
            k,
            dist: [k1, k2, p1, p2],
            resolution: (width, height),
            cv_k,
            cv_dist,
        }
    }
}

impl CameraModel for PinholeRadTan {
    /// 相机坐标系->像素坐标
    fn project(&self, point_in_c: &Vector3<f64>) -> Option<Vector2<f64>> {
        if point_in_c.z <= 0.0 {
            return None;
        }

        let z_inv = 1.0 / point_in_c.z;
        let xn = point_in_c.x * z_inv;
        let yn = point_in_c.y * z_inv;

        let r2 = xn * xn + yn * yn;
        let r4 = r2 * r2;

        let radial = 1.0 + self.dist[0] * r2 + self.dist[1] * r4;

        let dx = 2.0 * self.dist[2] * xn * yn + self.dist[3] * (r2 + 2.0 * xn * xn);
        let dy = self.dist[2] * (r2 + 2.0 * yn * yn) + 2.0 * self.dist[3] * xn * yn;

        let x_distorted = xn * radial + dx;
        let y_distorted = yn * radial + dy;

        Some(Vector2::new(
            self.k[(0, 0)] * x_distorted + self.k[(0, 2)],
            self.k[(1, 1)] * y_distorted + self.k[(1, 2)],
        ))
    }

    /// 反投影
    /// 像素坐标->相机坐标系
    fn unproject(&self, points: &Vector<Point2f>) -> Vec<Vector3<f64>> {
        let mut dst_vec = Vector::<Point2f>::new();

        calib3d::undistort_points(
            points,
            &mut dst_vec,
            &self.cv_k,
            &self.cv_dist,
            &Mat::default(),
            &Mat::default(),
        )
        .expect("OpenCV batch undistort failed");

        dst_vec
            .into_iter()
            .map(|p| Vector3::new(p.x as f64, p.y as f64, 1.0))
            .collect()
    }

    /// 投影雅可比矩阵,针孔部分的线性项,对去畸变后的点进行线性化
    fn projection_jacobian(&self, point_in_c: &Vector3<f64>) -> Matrix2x3<f64> {
        let x = point_in_c.x;
        let y = point_in_c.y;
        let z = point_in_c.z;
        let z2 = z * z;

        // 这是归一化平面投影相对于 3D 点的导数 (de/dP)
        // 如果后端处理的是归一化坐标，这样写是对的
        Matrix2x3::new(1.0 / z, 0.0, -x / z2, 0.0, 1.0 / z, -y / z2)
    }

    fn dimensions(&self) -> (usize, usize) {
        self.resolution
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;
    use opencv::core::{Point2f, Vector};

    /// 验证 PinholeRadTan 模型在 EuroC 数据集参数下的表现
    #[test]
    fn test_pinhole_radtan_euroc_params() {
        // 1. 初始化模型 (参数来源于你的 yaml)
        let cam = PinholeRadTan::new(
            458.654,
            457.296,
            367.215,
            248.375, // intrinsics
            -0.28340811,
            0.07395907,
            0.00019359,
            1.76187114e-05, // distortion
            752,
            480, // resolution
        );

        // 2. 测试投影：假设在相机正前方 2 米处有一个点 [0.5, 0.5, 2.0]
        let p_c = Vector3::new(0.5, 0.5, 2.0);
        let pixel = cam.project(&p_c).unwrap();

        println!("3D Point {:?} projected to pixel: {:?}", p_c, pixel);

        // 校验：点在图像范围内
        assert!(pixel.x >= 0.0 && pixel.x <= 752.0);
        assert!(pixel.y >= 0.0 && pixel.y <= 480.0);

        // 3. 测试去畸变一致性 (Round-trip test)
        // 选取一个典型的像素点 (例如图像中心偏一点)
        let raw_pixel_coord = Point2f::new(400.0, 300.0);
        let mut pixels_in = Vector::<Point2f>::new();
        pixels_in.push(raw_pixel_coord);

        // 反投影到归一化平面 (x, y, 1.0)
        let unprojected = cam.unproject(&pixels_in);
        let p_norm = unprojected[0];

        // 这里的 p_norm 应该是去畸变后的归一化坐标，我们将其重新投影
        // 注意：project 函数内部会重新应用畸变，所以结果应该回到 raw_pixel_coord
        let reprojected = cam.project(&p_norm).unwrap();

        println!("Original pixel: {:?}", raw_pixel_coord);
        println!("Reprojected pixel: {:?}", reprojected);

        // 允许微小的浮点数误差 (通常在 1e-6 像素以内)
        assert!((reprojected.x - raw_pixel_coord.x as f64).abs() < 1e-6);
        assert!((reprojected.y - raw_pixel_coord.y as f64).abs() < 1e-6);
    }
}
