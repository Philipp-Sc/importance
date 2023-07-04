use std::ops::Deref;
use crate::score::{Model, score, score_with_indices, ScoreKind};
use rand::prelude::SliceRandom;
use rand::thread_rng;

use rayon::prelude::*;
use std::sync::{Arc, Mutex};
pub mod score;


fn all_permutation_score(model: &dyn Model, x: Arc<Vec<Vec<f64>>>, y: &Vec<f64>, kind: ScoreKind, n_repeats: usize) -> f64 {
    let chunk_size = x[0].len();

    let scores: Vec<f64> = (0..n_repeats).into_par_iter().map_init(|| {
        let mut rng = thread_rng();
        let indices: Vec<usize> = (0..chunk_size).collect();
        (rng, indices)
    }, |(rng, indices), _| {
        indices.shuffle(rng);

        score_with_indices(model, &x, &indices[..], y, kind).unwrap()
    }).collect();

    scores.iter().sum::<f64>() / n_repeats as f64
}


pub fn permutation_scores(model: &dyn Model, x: Arc<Vec<Vec<f64>>>, y: &Vec<f64>, kind: ScoreKind, id: usize, n_repeats: usize) -> Vec<f64> {
    (0..n_repeats).into_par_iter().map_init(|| {
        let mut rng = thread_rng();
        let mut x = x.deref().clone();
        let mut column: Vec<f64> = x.iter().map(|row| row[id]).collect();
        (rng, x, column)
    }, |(rng, x, column), _| {
        column.shuffle(rng);
        for (row, &value) in x.iter_mut().zip(column.iter()) {
            row[id] = value;
        }
        score(model, &x, y, kind).unwrap()
    }).collect()
}



#[derive(Debug)]
pub struct ImportanceResult {
    pub importances: Vec<Vec<f64>>,
    pub importances_means: Vec<f64>,
    pub importances_stds: Vec<f64>,
}

pub struct Opts {
    pub verbose: bool,
    pub kind: Option<ScoreKind>,
    pub n: Option<usize>,
    pub only_means: bool,
    pub scale: bool,
}

pub fn importance(model: &dyn Model, x: Vec<Vec<f64>>, y: Vec<f64>, opts: Opts) -> ImportanceResult {
    let x = Arc::new(x);
    let base_score = score(model, &x, &y, opts.kind.unwrap()).unwrap();
    let n_features = x[0].len();

    let mut importances: Vec<Vec<f64>> = (0..n_features).into_par_iter()
        .map(|i| {
            let perm_scores = permutation_scores(model, x.clone(), &y, opts.kind.unwrap(), i, opts.n.unwrap());
            perm_scores.into_iter().map(|score| base_score - score).collect::<Vec<_>>()
        }).collect();

    if opts.scale {
        let perm_score = all_permutation_score(model, x.clone(), &y, opts.kind.unwrap(), opts.n.unwrap());
        let best_score = match opts.kind.unwrap() {
            ScoreKind::Acc => {100.0}
            _ => {0.0}
        };
        let factor = best_score - perm_score;
        importances = importances.iter().map(|imp| imp.iter().map(|&v| v / if factor!=0.0 {factor}else{1.0}).collect()).collect();
    }

    let importances_means = importances.iter().map(|imps| imps.iter().sum::<f64>() / opts.n.unwrap() as f64).collect::<Vec<_>>();



    if opts.only_means {
        ImportanceResult {
            importances: vec![],
            importances_means,
            importances_stds: vec![],
        }
    } else {
        let importances_stds: Vec<f64> = importances.iter().enumerate().map(|(i, imps)| {
            let mean = importances_means[i];
            (imps.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / opts.n.unwrap() as f64).sqrt()
        }).collect();

        ImportanceResult {
            importances,
            importances_means,
            importances_stds,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockModel;

    impl Model for MockModel {
        fn predict(&self, x: &Vec<Vec<f64>>) -> Vec<f64> {
            x.iter().map(|x| x.iter().sum()).collect() // Mock predictions
        }
    }

    #[test]
    fn it_works() {
        let model = MockModel;
        let x = vec![vec![1.0, 0.0, 3.0], vec![4.0, 0.0, 6.0], vec![7.0, 0.0, 9.0]];
        let y = vec![4.0, 10.0, 16.0];

        let opts = Opts {
            verbose: true,
            kind: Some(ScoreKind::Smape),
            n: Some(100),
            only_means: true,
            scale: true,
        };

        let importances = importance(&model, x, y, opts);
        println!("Importances: {:?}", importances);
    }
}
