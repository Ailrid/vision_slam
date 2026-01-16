// #[cfg(test)]
// mod coord_test {
//     use geodesy::prelude::*;
//     use std::time::Instant;

//     #[test]
//     fn main() -> anyhow::Result<()> {
//         // --- 准备阶段 ---
//         let start_total = Instant::now();
//         let mut ctx = Minimal::default();
//         let zone = 50;

//         // --- 1. 测试管道解析/创建耗时 ---
//         let t1 = Instant::now();
//         let pipeline = format!("inv utm zone={} | cart", zone);
//         let op = ctx.op(&pipeline)?;
//         let d1 = t1.elapsed();

//         // --- 2. 准备数据 ---
//         let utm_x = 442260.05812379945;
//         let utm_y = 4419080.5284244185;
//         let utm_z = 0.0;
//         let mut data = [Coor3D::raw(utm_x, utm_y, utm_z)];

//         // --- 3. 测试单次转换耗时 ---
//         let t2 = Instant::now();
//         ctx.apply(op, Fwd, &mut data)?;
//         let d2 = t2.elapsed();

//         // --- 4. 测试批量/多次转换耗时 (模拟你的 k=10 场景) ---
//         let t3 = Instant::now();
//         for _ in 0..10 {
//             let mut data_batch = [Coor3D::raw(utm_x, utm_y, utm_z)];
//             ctx.apply(op, Fwd, &mut data_batch)?;
//         }
//         let d3 = t3.elapsed();

//         let total_duration = start_total.elapsed();

//         // --- 输出结果 ---
//         let sep = "=".repeat(40);
//         println!("\n{}", sep);
//         println!("Geodesy 性能测试报告 (1Hz 场景)");
//         println!("{}", sep);
//         println!("1. 管道解析耗时 (Pipeline Parsing): {:?}", d1);
//         println!("2. 单次坐标转换耗时 (Single Apply): {:?}", d2);
//         println!("3. 10次循环转换耗时 (k=10 Apply):   {:?}", d3);
//         println!("4. 测试总耗时 (Total Context):     {:?}", total_duration);
//         println!("{}", "-".repeat(40));
        
//         let res = data[0];
//         println!("ECEF 结果 -> X: {:.3}, Y: {:.3}, Z: {:.3}", res[0], res[1], res[2]);
//         println!("{}", sep);

//         Ok(())
//     }
// }