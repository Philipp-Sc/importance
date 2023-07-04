# Permutation feature importance   


### Feature selection
Feature importance is often used for variable selection.  

### Compute the relative importance of input variables of trained predictive models using feature shuffling

When called, the `importance` function shuffles each feature `n` times and computes the difference between the base score (calculated with original features `X` and target variable `y`) and permuted data. Intuitively that measures how the performance of a model decreases when we "remove" the feature.

- More info about the method: [Permutation Feature Importance](https://christophm.github.io/interpretable-ml-book/feature-importance.html)
- Permutation importance can be biased if features are highly correlated ([Hooker,  Mentch 2019](https://arxiv.org/pdf/1905.03151v1.pdf))

### Usage
```rust
    struct MockModel;

    impl Model for MockModel {
        fn predict(&self, x: &Vec<Vec<f64>>) -> Vec<f64> {
            x.iter().map(|x| x.iter().sum()).collect() // Mock predictions
        }
    }
 
    fn main() {
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
```

### API
```rust
importance(model, X, y, options)
```
- `model` - trained model with `predict` method  
- `X` - 2D array of features
- `y` - 1D array of target variables

Options:
- `kind` - scoring function (`mse`, `mae`, `rmse`, `smape`, `acc`, `ce` (cross-entropy)
- `n` - number of times each feature is shuffled. 
- `only_means` - if `true` returns only average importance
- `verbose` - if `true` throws some info into console


 ### This is a rust port of https://github.com/zemlyansky/importance

