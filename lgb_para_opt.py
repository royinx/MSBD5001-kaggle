import skopt
from lgb3_byday import train_evaluate
# from lgb2_byhour import train_evaluate
SPACE = [
    skopt.space.Real(0.01, 0.5, name='learning_rate', prior='log-uniform'),
    skopt.space.Integer(1, 30, name='max_depth'),
    skopt.space.Integer(2, 100, name='num_leaves'),
    skopt.space.Integer(200, 500, name='num_trees'),
    skopt.space.Integer(200, 500, name='max_bin'),
    
    skopt.space.Real(0.1, 1.0, name='feature_fraction', prior='uniform'),
    skopt.space.Real(0.1, 1.0, name='subsample', prior='uniform')
    
    ]
@skopt.utils.use_named_args(SPACE)
def objective(**params):
    return train_evaluate(params)
results = skopt.forest_minimize(objective, SPACE, n_calls=30, n_random_starts=10)
best_mse = results.fun
best_params = results.x
print('best result (MSE): ', best_mse)
print('best parameters: ', best_params)

# dump all models with mse below 9.5