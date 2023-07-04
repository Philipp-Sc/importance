use std::error::Error;
use std::sync::Arc;

// Updated ScoreKind enum without Ce
#[derive(Clone, Copy)]
pub enum ScoreKind {
    Mae,
    Mse,
    Rmse,
    Smape,
    Acc,
}

pub trait Model: Send + Sync {
    fn predict(&self, x: &Vec<Vec<f64>>) -> Vec<f64>;

    fn predict_with_indices(&self, x: &Arc<Vec<Vec<f64>>>, indices: &[usize]) -> Vec<f64> {
        // implement this function without using .clone() to improve the performance further.
        let x_permutated: Vec<Vec<f64>> = indices.iter().map(|&i| x[i].clone()).collect();
        self.predict(&x_permutated)
    }
}

fn mae(yt: &Vec<f64>, yp: &Vec<f64>) -> f64 {
    yt.iter().zip(yp.iter()).map(|(a, b)| (a - b).abs()).sum::<f64>() / yt.len() as f64
}

fn mse(yt: &Vec<f64>, yp: &Vec<f64>) -> f64 {
    yt.iter().zip(yp.iter()).map(|(a, b)| (a - b).powf(2.0)).sum::<f64>() / yt.len() as f64
}

fn rmse(yt: &Vec<f64>, yp: &Vec<f64>) -> f64 {
    (mse(yt, yp)).sqrt()
}

fn smape(yt: &Vec<f64>, yp: &Vec<f64>) -> f64 {
    let sum = yt.iter().zip(yp.iter()).map(|(a, b)| 2.0 * (a - b).abs() / (a.abs() + b.abs())).sum::<f64>();
    (sum / yt.len() as f64) * 100.0
}

fn acc(yt: &Vec<f64>, yp: &Vec<f64>) -> f64 {
    yt.iter().zip(yp.iter()).map(|(a, b)| if a == b { 1.0 } else { 0.0 }).sum::<f64>() / yt.len() as f64
}

pub fn score(model: &dyn Model, x: &Vec<Vec<f64>>, y: &Vec<f64>, kind: ScoreKind) -> Result<f64, &'static str> {
    if y.len() != x.len() {
        return Err("Arrays have different length");
    }
    if y.is_empty() {
        return Err("Zero length array");
    }
    let yp = model.predict(&x);

    let score = match kind {
        ScoreKind::Mae => mae(y, &yp),
        ScoreKind::Mse => mse(y, &yp),
        ScoreKind::Rmse => rmse(y, &yp),
        ScoreKind::Smape => smape(y, &yp),
        ScoreKind::Acc => acc(y, &yp),
    };
    Ok(score)
}

pub fn score_with_indices(
    model: &dyn Model,
    x_arc: &Arc<Vec<Vec<f64>>>,
    indices: &[usize],
    y: &Vec<f64>,
    kind: ScoreKind,
) -> Result<f64, Box<dyn Error>> {
    let prediction = model.predict_with_indices(&x_arc, indices);
    Ok(match kind {
        ScoreKind::Mae => mae(&prediction, y),
        ScoreKind::Mse => mse(&prediction, y),
        ScoreKind::Rmse => rmse(&prediction, y),
        ScoreKind::Smape => smape(&prediction, y),
        ScoreKind::Acc => acc(&prediction, y),
    })
}




#[cfg(test)]
mod tests {
    use super::*;

    struct MockModel;

    impl Model for MockModel {
        fn predict(&self, _x: &Vec<Vec<f64>>) -> Vec<f64> {
            vec![0.4, 0.6, 0.8] // Mock predictions
        }
    }

    #[test]
    fn it_works() {
        let model = MockModel;
        let x =  vec![vec![],vec![],vec![]];
        let y = vec![0.4, 0.6, 0.8];

        let expected_mae_score = 0.0;
        let mae_score = score(&model, &x, &y, ScoreKind::Mae).unwrap();
        println!("MAE: {}", mae_score);
        assert_eq!(mae_score, expected_mae_score, "MAE Score does not match");

        let expected_mse_score = 0.0;
        let mse_score = score(&model, &x, &y, ScoreKind::Mse).unwrap();
        println!("MSE: {}", mse_score);
        assert_eq!(mse_score, expected_mse_score, "MSE Score does not match");

        let expected_rmse_score = 0.0;
        let rmse_score = score(&model, &x, &y, ScoreKind::Rmse).unwrap();
        println!("RMSE: {}", rmse_score);
        assert_eq!(rmse_score, expected_rmse_score, "RMSE Score does not match");

        let expected_smape_score = 0.0;
        let smape_score = score(&model, &x, &y, ScoreKind::Smape).unwrap();
        println!("SMAPE: {}", smape_score);
        assert_eq!(smape_score, expected_smape_score, "SMAPE Score does not match");

        let expected_acc_score = 1.0;
        let acc_score = score(&model, &x, &y, ScoreKind::Acc).unwrap();
        println!("Accuracy: {}", acc_score);
        assert_eq!(acc_score, expected_acc_score, "Accuracy Score does not match");


    }
}
