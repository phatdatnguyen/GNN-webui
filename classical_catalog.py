"""Supported scikit-learn estimators, editable parameters, and safe fitted pipelines.

The public estimator APIs used here are available in scikit-learn 1.4 and later.
Probability calibration wraps the complete feature-scaling pipeline, so every
calibration fold fits its own preprocessing using only that fold's training rows.
"""

from copy import deepcopy
import inspect

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    AdaBoostClassifier, AdaBoostRegressor, BaggingClassifier, BaggingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier,
    GradientBoostingRegressor, HistGradientBoostingClassifier,
    HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, RationalQuadratic
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (
    ARDRegression, BayesianRidge, ElasticNet, GammaRegressor, HuberRegressor,
    Lasso, LinearRegression, LogisticRegression, OrthogonalMatchingPursuit,
    Perceptron, PoissonRegressor, QuantileRegressor, RANSACRegressor, Ridge,
    RidgeClassifier, SGDClassifier, SGDRegressor, TweedieRegressor,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC, LinearSVR, SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor


def _number(kind, default, minimum, maximum):
    return {"type": kind, "default": default, "min": minimum, "max": maximum}


def _choice(default, *choices):
    return {"type": "choice", "default": default, "choices": list(choices)}


MODEL_PARAMETERS = {"classification": {}, "regression": {}}
MODEL_DESCRIPTIONS = {"classification": {}, "regression": {}}
_ESTIMATORS = {"classification": {}, "regression": {}}
_SCALED = set()
_CALIBRATED = {"SVC", "Linear SVC", "Ridge Classifier", "Perceptron"}
_QDA_EIGEN = "solver" in inspect.signature(QuadraticDiscriminantAnalysis).parameters


def _register(task, name, estimator, parameters, description, *, scaled=False):
    _ESTIMATORS[task][name] = estimator
    MODEL_PARAMETERS[task][name] = deepcopy(parameters)
    MODEL_DESCRIPTIONS[task][name] = description
    if scaled:
        _SCALED.add((task, name))


_CLASS_WEIGHT = _choice("None", "None", "balanced")
_FIT_INTERCEPT = _choice("True", "True", "False")
_TREE = {"max_depth": _number("int", 0, 0, 100),
         "min_samples_leaf": _number("int", 1, 1, 1000)}
_FOREST = {"n_estimators": _number("int", 100, 1, 2000), **_TREE}
_BOOSTING = {"n_estimators": _number("int", 100, 1, 2000),
             "learning_rate": _number("float", 0.1, 0.0001, 1.0),
             "max_depth": _number("int", 3, 1, 100)}
_SVM = {"C": _number("float", 1.0, 0.0001, 10000),
        "kernel": _choice("rbf", "rbf", "linear", "poly", "sigmoid"),
        "gamma": _choice("scale", "scale", "auto")}
_NEIGHBORS = {"n_neighbors": _number("int", 5, 1, 1000),
              "weights": _choice("uniform", "uniform", "distance")}

# Preserve the original six model names/order and defaults in each task.
for task, forest, extra, boosting, neighbors in [
    ("classification", RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, KNeighborsClassifier),
    ("regression", RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, KNeighborsRegressor),
]:
    forest_parameters = {**_FOREST, **({"class_weight": _CLASS_WEIGHT} if task == "classification" else {})}
    _register(task, "Random Forest", forest, forest_parameters,
              "A nonlinear baseline using many decision trees. More trees increase fit time; 0 maximum depth means unlimited.")
    _register(task, "Extra Trees", extra, forest_parameters,
              "An ensemble of randomized decision trees. More trees increase fit time; 0 maximum depth means unlimited.")
    _register(task, "Gradient Boosting", boosting, _BOOSTING,
              "Fits shallow trees sequentially. Tune learning rate, tree depth, and number of trees together.")

_register("classification", "Logistic Regression", LogisticRegression,
          {"C": _number("float", 1.0, 0.0001, 10000),
           "max_iter": _number("int", 1000, 1, 10000), "class_weight": _CLASS_WEIGHT},
          "A regularized linear probability model. Features are standardized using training data only.", scaled=True)
_register("classification", "SVC", SVC, {**_SVM, "class_weight": _CLASS_WEIGHT},
          "A kernel classifier with probabilities calibrated inside the training split. Requires at least 2 training examples of each class per target; can be slow on large datasets.", scaled=True)
_register("regression", "Ridge", Ridge, {"alpha": _number("float", 1.0, 0, 10000)},
          "Linear regression with L2 regularization and training-fitted feature scaling.", scaled=True)
_register("regression", "SVR", SVR, {**_SVM, "epsilon": _number("float", 0.1, 0, 1000)},
          "Kernel regression with an epsilon-insensitive loss. Features are standardized; fitting can be slow on large datasets.", scaled=True)
for task, estimator in [("classification", KNeighborsClassifier), ("regression", KNeighborsRegressor)]:
    _register(task, "K-Nearest Neighbors", estimator, _NEIGHBORS,
              "Predicts from nearby training fingerprints after feature scaling. Neighbor count must not exceed the training split size.", scaled=True)

_HIST = {"learning_rate": _number("float", 0.1, 0.0001, 1.0),
         "max_iter": _number("int", 100, 1, 2000),
         "max_leaf_nodes": _number("int", 31, 2, 255),
         "min_samples_leaf": _number("int", 10, 1, 1000),
         "l2_regularization": _number("float", 0.0, 0, 10000)}
_ADABOOST = {"n_estimators": _number("int", 50, 1, 2000),
             "learning_rate": _number("float", 1.0, 0.0001, 10),
             "max_depth": _number("int", 1, 1, 100)}
_BAGGING = {"n_estimators": _number("int", 50, 1, 2000), **_TREE,
            "max_samples": _number("float", 1.0, 0.01, 1.0)}
for task, tree, extra, hist, ada, bag in [
    ("classification", DecisionTreeClassifier, ExtraTreeClassifier, HistGradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier),
    ("regression", DecisionTreeRegressor, ExtraTreeRegressor, HistGradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor),
]:
    tree_parameters = {**_TREE, **({"class_weight": _CLASS_WEIGHT} if task == "classification" else {})}
    _register(task, "Decision Tree", tree, tree_parameters,
              "A single decision tree. Limit depth or increase minimum leaf size to reduce overfitting.")
    _register(task, "Extra Tree", extra, tree_parameters,
              "A single tree with randomized split thresholds. Limit depth or increase minimum leaf size to reduce overfitting.")
    _register(task, "Histogram Gradient Boosting", hist, _HIST,
              "Boosted trees trained with feature histograms. All selected training rows are used; early stopping is disabled.")
    _register(task, "AdaBoost", ada, _ADABOOST,
              "Boosts shallow decision trees by emphasizing difficult examples. Maximum depth controls each base tree.")
    _register(task, "Bagging", bag, _BAGGING,
              "Fits independent decision trees on bootstrap samples. Maximum samples is the fraction used by each tree.")

_register("classification", "Gaussian Naive Bayes", GaussianNB,
          {"var_smoothing": _number("float", 1e-9, 1e-12, 1.0)},
          "A fast conditional-independence baseline with Gaussian feature distributions. Increase variance smoothing for nearly constant features.")
for name, estimator in [("Bernoulli Naive Bayes", BernoulliNB),
                        ("Multinomial Naive Bayes", MultinomialNB),
                        ("Complement Naive Bayes", ComplementNB)]:
    _register("classification", name, estimator,
              {"alpha": _number("float", 1.0, 1e-10, 1000)},
              "A fast conditional-independence baseline on unscaled fingerprint bits. Multinomial and complement variants require nonnegative features.")
_register("classification", "Linear Discriminant Analysis", LinearDiscriminantAnalysis,
          {"tol": _number("float", 1e-4, 1e-10, 0.1)},
          "Uses a shared covariance model and an SVD solver. Requires variation within the training classes; correlated fingerprints can reduce effective rank.", scaled=True)
_register("classification", "Quadratic Discriminant Analysis", QuadraticDiscriminantAnalysis,
          {"shrinkage" if _QDA_EIGEN else "reg_param": _number("float", 0.1, 0.0, 1.0)},
          "Uses separate class covariance models. Requires at least 2 training examples per class; covariance regularization helps with correlated features. Wide fingerprint covariance matrices can be expensive.", scaled=True)
_register("classification", "SGD Classifier", SGDClassifier,
          {"alpha": _number("float", 0.0001, 1e-8, 10),
           "penalty": _choice("l2", "l2", "l1", "elasticnet"),
           "max_iter": _number("int", 1000, 1, 10000)},
          "A scalable linear classifier trained with stochastic gradient descent and logistic loss. Training-only feature scaling is included.", scaled=True)
_register("classification", "Linear SVC", LinearSVC,
          {"C": _number("float", 1.0, 0.0001, 10000),
           "max_iter": _number("int", 2000, 1, 20000), "class_weight": _CLASS_WEIGHT},
          "A linear support-vector classifier with training-only probability calibration. Requires at least 2 training examples of each class per target.", scaled=True)
_register("classification", "Ridge Classifier", RidgeClassifier,
          {"alpha": _number("float", 1.0, 0, 10000), "class_weight": _CLASS_WEIGHT},
          "A regularized linear classifier with training-only probability calibration. Requires at least 2 training examples of each class per target.", scaled=True)
_register("classification", "Perceptron", Perceptron,
          {"alpha": _number("float", 0.0001, 1e-8, 10),
           "penalty": _choice("l2", "l2", "l1", "elasticnet"),
           "max_iter": _number("int", 1000, 1, 10000), "class_weight": _CLASS_WEIGHT},
          "An online linear classifier with training-only probability calibration. Requires at least 2 training examples of each class per target.", scaled=True)

_register("regression", "Linear Regression", LinearRegression,
          {"fit_intercept": _FIT_INTERCEPT},
          "Ordinary least-squares linear regression with training-only feature scaling.", scaled=True)
_LINEAR_REGULARIZED = {"alpha": _number("float", 1.0, 1e-8, 10000),
                       "max_iter": _number("int", 1000, 1, 20000)}
_register("regression", "Lasso", Lasso, _LINEAR_REGULARIZED,
          "Linear regression with L1 regularization, which can set irrelevant fingerprint coefficients to zero.", scaled=True)
_register("regression", "Elastic Net", ElasticNet,
          {**_LINEAR_REGULARIZED, "l1_ratio": _number("float", 0.5, 0.01, 1.0)},
          "Combines L1 and L2 regularization for correlated fingerprints; uses training-only feature scaling.", scaled=True)
for name, estimator in [("Bayesian Ridge", BayesianRidge), ("ARD Regression", ARDRegression)]:
    _register("regression", name, estimator,
              {"max_iter": _number("int", 300, 1, 5000), "tol": _number("float", 0.001, 1e-8, 0.1)},
              "Bayesian linear regression with training-only feature scaling. ARD learns feature-specific regularization and can be expensive with thousands of fingerprint bits.", scaled=True)
_register("regression", "Huber Regression", HuberRegressor,
          {"epsilon": _number("float", 1.35, 1.0, 10.0),
           "alpha": _number("float", 0.0001, 0, 10000), "max_iter": _number("int", 300, 1, 10000)},
          "Robust linear regression that reduces the influence of target outliers. Features are standardized using training rows.", scaled=True)
_register("regression", "SGD Regressor", SGDRegressor,
          {"alpha": _number("float", 0.0001, 1e-8, 10),
           "loss": _choice("squared_error", "squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"),
           "penalty": _choice("l2", "l2", "l1", "elasticnet"),
           "max_iter": _number("int", 1000, 1, 20000)},
          "Scalable linear regression trained with stochastic gradient descent and training-only feature scaling.", scaled=True)
_GLM = {"alpha": _number("float", 1.0, 0, 10000),
        "max_iter": _number("int", 300, 1, 10000)}
_register("regression", "Poisson Regression", PoissonRegressor, _GLM,
          "A log-link model for nonnegative targets. Training targets must be nonnegative and include a positive value.", scaled=True)
_register("regression", "Gamma Regression", GammaRegressor, _GLM,
          "A log-link model for strictly positive continuous targets. Every training target must be greater than zero.", scaled=True)
_register("regression", "Tweedie Regression", TweedieRegressor,
          {**_GLM, "power": _choice("1.5", "0", "1", "1.5", "2", "3")},
          "Power 0 allows real targets; powers 1 and 1.5 require nonnegative targets with a positive value; powers 2 and 3 require strictly positive targets.", scaled=True)
_register("regression", "Kernel Ridge", KernelRidge,
          {"alpha": _number("float", 1.0, 1e-8, 10000),
           "kernel": _choice("rbf", "rbf", "linear", "poly", "laplacian", "sigmoid", "cosine"),
           "gamma": _number("float", 0.0, 0, 10000), "degree": _number("int", 3, 1, 10)},
          "Ridge regression in kernel space. Gamma 0 uses 1 / feature count. Memory grows quadratically with the number of training molecules.", scaled=True)
_register("regression", "Linear SVR", LinearSVR,
          {"C": _number("float", 1.0, 0.0001, 10000),
           "epsilon": _number("float", 0.0, 0, 1000), "max_iter": _number("int", 2000, 1, 20000)},
          "Linear support-vector regression with training-only feature scaling; usually faster than kernel SVR on large datasets.", scaled=True)
_register("regression", "Orthogonal Matching Pursuit", OrthogonalMatchingPursuit,
          {"n_nonzero_coefs": _number("int", 0, 0, 4096)},
          "Greedily selects a sparse linear set of fingerprint features. Nonzero coefficients 0 uses sklearn's default; the count cannot exceed feature width.", scaled=True)
_register("regression", "RANSAC Regression", RANSACRegressor,
          {"min_samples": _number("float", 0.5, 0.01, 1.0),
           "max_trials": _number("int", 100, 1, 10000),
           "residual_threshold": _number("float", 0.0, 0, 1000000)},
          "Robust linear regression fitted to consensus subsets. Minimum samples is a fraction; residual threshold 0 uses the median absolute deviation.", scaled=True)
_register("regression", "Quantile Regression", QuantileRegressor,
          {"quantile": _number("float", 0.5, 0.01, 0.99), "alpha": _number("float", 1.0, 0, 10000)},
          "Fits a chosen conditional target quantile using linear programming. Large, wide fingerprint matrices can require substantial computation.", scaled=True)

_MLP = {"hidden_layer_sizes": _choice("100", "50", "100", "200", "100,50", "200,100"),
        "activation": _choice("relu", "relu", "tanh", "logistic"),
        "solver": _choice("adam", "adam", "lbfgs", "sgd"),
        "alpha": _number("float", 0.0001, 0, 10000),
        "learning_rate_init": _number("float", 0.001, 1e-6, 1.0),
        "max_iter": _number("int", 300, 1, 10000)}
_GP = {"kernel": _choice("RBF", "RBF", "Matern", "Rational Quadratic"),
       "length_scale": _number("float", 1.0, 0.001, 1000),
       "optimizer": _choice("None", "None", "fmin_l_bfgs_b")}
for task, mlp, gp in [("classification", MLPClassifier, GaussianProcessClassifier),
                       ("regression", MLPRegressor, GaussianProcessRegressor)]:
    _register(task, "Multilayer Perceptron", mlp, _MLP,
              "A feed-forward neural network on fingerprints. Features are standardized; increase iterations if fitting does not converge.", scaled=True)
    gp_parameters = {**_GP, **({"max_iter_predict": _number("int", 100, 1, 1000)}
                               if task == "classification" else {
                                   "alpha": _number("float", 1e-6, 1e-12, 1000),
                                   "normalize_y": _choice("True", "True", "False")})}
    _register(task, "Gaussian Process", gp, gp_parameters,
              "A flexible kernel model best suited to small datasets: memory grows quadratically and fitting roughly cubically with sample count. Kernel optimization adds fitting cost.", scaled=True)
_register("classification", "Dummy Classifier", DummyClassifier,
          {"strategy": _choice("prior", "prior", "most_frequent", "stratified", "uniform")},
          "A reference baseline that ignores fingerprint features. Compare learned models against this baseline.")
_register("regression", "Dummy Regressor", DummyRegressor,
          {"strategy": _choice("mean", "mean", "median", "quantile"),
           "quantile": _number("float", 0.5, 0.01, 0.99)},
          "A reference baseline that ignores fingerprint features and predicts a training target summary.")

MODEL_NAMES = {task: tuple(models) for task, models in MODEL_PARAMETERS.items()}
MODEL_NOTES = MODEL_DESCRIPTIONS


def validate_prediction_data(task, model_name, X):
    """Apply feature-domain restrictions to both fitted-model inference and training."""
    if (task == "classification" and model_name in {"Multinomial Naive Bayes", "Complement Naive Bayes"}
            and np.any(X < 0)):
        raise ValueError(f"{model_name} requires nonnegative fingerprint features.")


def validate_training_data(task, model_name, parameters, X, y):
    """Check model-specific training requirements before fitting any target estimator."""
    y = np.asarray(y)
    validate_prediction_data(task, model_name, X)
    if task == "classification":
        _, counts = np.unique(y, return_counts=True)
        if model_name in _CALIBRATED and (len(counts) != 2 or counts.min() < 2):
            raise ValueError(f"{model_name} probability calibration requires at least 2 training "
                             "examples of each class. Change the split or choose another model.")
        if model_name == "Quadratic Discriminant Analysis" and counts.min() < 2:
            raise ValueError("Quadratic Discriminant Analysis requires at least 2 training examples per class.")
        if model_name == "Linear Discriminant Analysis":
            if len(y) <= len(counts) or not any(np.var(X[y == label], axis=0).max() > 0 for label in np.unique(y)):
                raise ValueError("Linear Discriminant Analysis requires feature variation within training classes.")
        if model_name == "Gaussian Naive Bayes" and not np.var(X, axis=0).max() > 0:
            raise ValueError("Gaussian Naive Bayes requires at least one varying fingerprint feature.")
    else:
        power = (float(parameters.get("power", 1.5)) if model_name == "Tweedie Regression"
                 else 1 if model_name == "Poisson Regression" else 2 if model_name == "Gamma Regression" else 0)
        if power >= 2 and np.any(y <= 0):
            raise ValueError(f"{model_name} requires strictly positive training targets (greater than zero).")
        if 1 <= power < 2 and (np.any(y < 0) or not np.any(y > 0)):
            raise ValueError(f"{model_name} requires nonnegative training targets with at least one positive value.")
        if model_name == "RANSAC Regression" and len(y) < 2:
            raise ValueError("RANSAC Regression requires at least 2 training molecules.")
        if model_name == "Orthogonal Matching Pursuit" and parameters.get("n_nonzero_coefs", 0) > X.shape[1]:
            raise ValueError("The requested nonzero coefficient count exceeds the fingerprint feature count.")
    if model_name == "K-Nearest Neighbors" and parameters["n_neighbors"] > len(y):
        raise ValueError("The neighbor count exceeds the training split size.")
    if model_name == "Bagging" and int(parameters["max_samples"] * len(y)) < 1:
        raise ValueError("Maximum samples selects no training molecules. Increase the fraction or training split.")


def make_pipeline(task, model_name, normalized_parameters, random_seed, *, y_train=None):
    """Construct an unfitted estimator, with preprocessing inside calibration folds."""
    options = dict(normalized_parameters)
    estimator_type = _ESTIMATORS[task][model_name]
    constructor_parameters = inspect.signature(estimator_type).parameters
    if "random_state" in constructor_parameters:
        options["random_state"] = random_seed
    if "n_jobs" in constructor_parameters and model_name != "Logistic Regression":
        options["n_jobs"] = 1
    for name in ("fit_intercept", "normalize_y"):
        if name in options and isinstance(options[name], str):
            options[name] = options[name] == "True"
    if model_name in {"AdaBoost", "Bagging"}:
        tree_type = DecisionTreeClassifier if task == "classification" else DecisionTreeRegressor
        tree_options = {"max_depth": options.pop("max_depth"), "random_state": random_seed}
        if model_name == "Bagging":
            tree_options["min_samples_leaf"] = options.pop("min_samples_leaf")
        options["estimator"] = tree_type(**tree_options)
        if task == "classification" and model_name == "AdaBoost" and "algorithm" in constructor_parameters:
            options["algorithm"] = "SAMME"
    if model_name == "Histogram Gradient Boosting":
        options["early_stopping"] = False
    if model_name == "SGD Classifier":
        options["loss"] = "log_loss"
    if model_name == "Quadratic Discriminant Analysis" and _QDA_EIGEN:
        options["solver"] = "eigen"
    if model_name == "Multilayer Perceptron":
        options["hidden_layer_sizes"] = tuple(int(width) for width in options["hidden_layer_sizes"].split(","))
        options["early_stopping"] = False
    if model_name == "Gaussian Process":
        kernel_name, length_scale = options.pop("kernel"), options.pop("length_scale")
        options["kernel"] = {"RBF": RBF, "Matern": Matern, "Rational Quadratic": RationalQuadratic}[kernel_name](length_scale=length_scale)
        if options["optimizer"] == "None":
            options["optimizer"] = None
    if model_name == "Tweedie Regression":
        options["power"] = float(options["power"])
    if model_name == "Kernel Ridge" and options["gamma"] == 0:
        options["gamma"] = None
    if model_name == "Orthogonal Matching Pursuit" and options["n_nonzero_coefs"] == 0:
        options["n_nonzero_coefs"] = None
    if model_name == "RANSAC Regression" and options["residual_threshold"] == 0:
        options["residual_threshold"] = None
    steps = [("scaler", StandardScaler())] if (task, model_name) in _SCALED else []
    if model_name in {"Dummy Classifier", "Dummy Regressor"}:
        # Older sklearn Dummy estimators do not record n_features_in_. An identity
        # transformer preserves schema validation without fitting a feature scaler.
        steps = [("schema", FunctionTransformer(validate=True))]
    pipeline = Pipeline([*steps, ("estimator", estimator_type(**options))])
    if task == "classification" and model_name in _CALIBRATED:
        if y_train is None:
            raise ValueError("Training labels are required to choose calibration folds.")
        _, counts = np.unique(y_train, return_counts=True)
        if len(counts) != 2 or counts.min() < 2:
            raise ValueError("Probability calibration requires at least 2 training examples of each class.")
        folds = StratifiedKFold(n_splits=min(3, int(counts.min())), shuffle=True, random_state=random_seed)
        return CalibratedClassifierCV(estimator=pipeline, cv=folds, method="sigmoid", ensemble=False, n_jobs=1)
    return pipeline


def validate_pipeline(task, model_name, pipeline):
    """Validate checkpoint estimator type and preprocessing structure without refitting."""
    if task == "classification" and model_name in _CALIBRATED:
        # Accept original SVC checkpoints, which used SVC(probability=True) directly.
        legacy_svc = (model_name == "SVC" and isinstance(pipeline, Pipeline)
                      and isinstance(pipeline.named_steps.get("estimator"), SVC)
                      and pipeline.named_steps["estimator"].probability)
        if not legacy_svc:
            if not isinstance(pipeline, CalibratedClassifierCV) or pipeline.ensemble is not False:
                raise ValueError("The classifier requires its fitted probability-calibration wrapper.")
            pipeline = pipeline.estimator
    expected_steps = ["scaler", "estimator"] if (task, model_name) in _SCALED else ["estimator"]
    if model_name in {"Dummy Classifier", "Dummy Regressor"}:
        expected_steps = ["schema", "estimator"]
    if (not isinstance(pipeline, Pipeline) or [name for name, _ in pipeline.steps] != expected_steps
            or not isinstance(pipeline.named_steps.get("estimator"), _ESTIMATORS[task][model_name])
            or ((task, model_name) in _SCALED and not isinstance(pipeline.named_steps["scaler"], StandardScaler))):
        raise ValueError("The saved estimator or preprocessing steps do not match the selected model.")
    if (model_name in {"Dummy Classifier", "Dummy Regressor"}
            and (not isinstance(pipeline.named_steps["schema"], FunctionTransformer)
                 or not pipeline.named_steps["schema"].validate
                 or pipeline.named_steps["schema"].func is not None)):
        raise ValueError("The baseline model must retain its fingerprint schema validator.")
