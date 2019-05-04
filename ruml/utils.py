import sklearn
from sklearn.linear_model import LinearRegression
import catboost
import pandas as pd
import copy
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
import inspect
import numpy as np
import skopt
import datetime
import os
from sklearn.externals import joblib
jl_compress = 3
def hello_world():
	print("HW!")

# Bayes Search EXample

# opt = skopt.BayesSearchCV(lgb.LGBMRegressor(n_estimators=3000, observation="mae"),
#           search_spaces= ruml.utils.SKOPT_BO["lgb"], verbose=True,
#             n_iter=3000,n_jobs=5,cv=folds, scoring="neg_mean_absolute_error",
#                          fit_params ={"early_stopping_rounds":200,"eval_set":[(X_early_stop,y_early_stop)]} )

# opt.fit(X=X_train,y=y_train, callback=[ruml.utils.Print_Callback(), skopt.callbacks.DeadlineStopper(total_time=36000)])



    

DEFAULT_VALUES = {

    "lgb": {
        "regr": lgb.LGBMRegressor,
        "model_params": { 
          "n_estimators":2000   
        },
        "fit_params":
        {
        "eval_metric":"mae",
        "early_stopping_rounds":150,
        "verbose":False
        }
        
        
    },
    
    "metrics":{
    "mae": mean_absolute_error,
        "r2": r2_score
    }
  
}


def conv_pred(preds):
    if (isinstance(preds,np.ndarray) and isinstance(preds[0],np.ndarray)) or (
        isinstance(preds,pd.Series) and isinstance(preds.iloc[0],np.ndarray)):
        preds = preds[:,0]
    return preds

def add_def_params(model_name, model_params, fit_params, def_param = DEFAULT_VALUES):
    
    if model_name in DEFAULT_VALUES.keys():
        if "model_params" in DEFAULT_VALUES[model_name]:
            new_p = copy.deepcopy(DEFAULT_VALUES[model_name]["model_params"])
            new_p.update(model_params)
            model_params = new_p
        if "fit_params" in DEFAULT_VALUES[model_name]:
            new_p = copy.deepcopy(DEFAULT_VALUES[model_name]["fit_params"])
            new_p.update(fit_params)
            fit_params = new_p
    return model_params, fit_params
                                  

    #model can be str, 
    #instance of estiomator - we use parameters of these estimator and model_params together
    #or estimator type we use model_params

def cv(model=LinearRegression, X = pd.DataFrame([]), y = pd.Series([]), folds = 5, 
      model_params = {},
      fit_params = {},
     task = "regr",
      metrics=["mae"]):
    model_name = None
    if isinstance(model,str):
        model_name = model
        model_params, fit_params = add_def_params(model, model_params,fit_params)
        model =  DEFAULT_VALUES[model_name][task]
    if not isinstance(model, type):
        model_params.update(model.get_params())
        model = type(model)
    predictions_cv = pd.Series([0]*len(X), index = X.index)
    predictions_cv_best = pd.Series([0]*len(X), index = X.index)
    scores = list()
    scores_best = list()
    models = list()
    best_iterations = list()
    if folds == 0:
        model_instance = model(**model_params)
        if "early_stopping_rounds" in fit_params.keys():
            fit_params = {k:v for k,v in fit_params.items() if k != "early_stopping_rounds"}
        model_instance = model_instance.fit( X, y, 
                   **fit_params)
        return {"models":[model_instance], "scores":[], "predictions_cv":None, "score_cv":None, "best_iterations": None}
    if  isinstance(folds,int):
        folds = KFold(n_splits=folds, shuffle=True, random_state=42)
       
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):
        X_train, X_valid = X.loc[train_index], X.loc[valid_index]
        y_train, y_valid = y.loc[train_index], y.loc[valid_index]
        model_instance = model(**model_params)
        if "eval_set" in inspect.getfullargspec( model_instance.fit).args:
            fit_params['eval_set'] = [(X_valid,y_valid)]
            
        model_instance.fit( X_train, y_train, 
                   **fit_params)
        train_predict =  conv_pred(model_instance.predict(X_train))
        train_score = list()
        for  metric in metrics:
            if  isinstance(metric,str):
                metric = DEFAULT_VALUES["metrics"][metric]
                train_score.append(metric(y_train,train_predict))        
        
        valid_predict = conv_pred(model_instance.predict(X_valid))
        predictions_cv[valid_index] = pd.Series(valid_predict)
        score = list()
        for  metric in metrics:
            if  isinstance(metric,str):
                metric = DEFAULT_VALUES["metrics"][metric]
                score.append(metric(y_valid,valid_predict))
        scores.append(score)
        models.append(model_instance)
        if hasattr(model_instance, "best_iteration_" ):
            best_iterations.append(model_instance.best_iteration_)
        print("Fold ", fold_n, "score  ", scores[-1], "train_score", train_score)

        if hasattr(model_instance,"predict_best"):
            valid_predict_best = model_instance.predict_best(X_valid, y_valid)
            valid_predict_best = conv_pred(valid_predict_best)
            predictions_cv_best[valid_index] = pd.Series(valid_predict_best)
            score_best = list()
            for  metric in metrics:
                if  isinstance(metric,str):
                    metric = DEFAULT_VALUES["metrics"][metric]
                    score_best.append(metric(y_valid,valid_predict_best))
            scores_best.append(score_best)
            print("score best  ", scores_best[-1], "\n")
         
        if hasattr(model_instance,"get_cluster_models"):
            clust_models = model_instance.get_cluster_models()
            for i, model_cluster in clust_models.items():
                valid_predict_model = conv_pred(model_cluster.predict(X_valid))
                score_model = list()
                for  metric in metrics:
                    if  isinstance(metric,str):
                        metric = DEFAULT_VALUES["metrics"][metric]
                        score_model.append(metric(y_valid,valid_predict_model))
                print("score best for model  ", i, " ", score_model, "\n")
                                  
                                  
                                  
                                  
        print("#"*30)
            
            
    score = list()
    for  metric in metrics:
        if  isinstance(metric,str):
            metric = DEFAULT_VALUES["metrics"][metric]
            score.append(metric(y,predictions_cv))
    print("Final scores: ", score)

    score_best = list()
    if len(scores_best)>0:
         for  metric in metrics:
            if  isinstance(metric,str):
                metric = DEFAULT_VALUES["metrics"][metric]
                score_best.append(metric(y,predictions_cv_best))
            print("Final scores best: ", score_best)

    return {"models":models, "scores":scores, "predictions_cv":predictions_cv, "score_cv":score,"score_cv_best":score_best, "best_iterations": best_iterations,
           "scores_best":scores_best, "model":model, "model_params":model_params, "fit_params":fit_params}

def blend_models(models,X):
    res = pd.Series([0]*len(X), index = X.index)
    for m in models:
        preds = m.predict(X)
        preds = conv_pred(preds)
        res+=preds
    res/=len(models)
    return res

#test
#ruml.utils.cv(X = pd.DataFrame({1:[i for i in range(10)],2:[2*i for i in range(10)]}),
#              y = pd.Series(i*i for i in range(10)),
#            )
 

lgbm_bo = {
            'num_leaves': (6, 1024),
          #  'max_depth': (4, 20),
            'learning_rate': (0.00001, 0.1),
            'bagging_fraction': (0.1, 1.0),
            'feature_fraction': (0.1, 1.0),
            'min_data_in_leaf': (6, 200),
            'bagging_freq': (0, 10),
            'reg_alpha': (0,100),
            'reg_lambda': (0,100),
        }


#           space.Integer(6, 30, name='num_leaves'),
#           space.Integer(50, 200, name='min_child_samples'),
#           space.Real(1, 400,  name='scale_pos_weight'),
#           space.Real(0.6, 0.9, name='subsample'),
#           space.Real(0.6, 0.9, name='colsample_bytree')

#objectives

# regression_l2, L2 loss, aliases: regression, mean_squared_error, mse, l2_root, root_mean_squared_error, rmse
# regression_l1, L1 loss, aliases: mean_absolute_error, mae
# huber, Huber loss
# fair, Fair loss
# poisson, Poisson regression
# quantile, Quantile regression
# mape, MAPE loss, aliases: mean_absolute_percentage_error
# gamma, Gamma regression with log-link. It might be useful, e.g., for modeling insurance claims severity, or for any target that might be gamma-distributed
# tweedie

SKOPT_BO = {
    
    "lgb" : {
         'num_leaves':  skopt.space.Integer(6, 512),
        'min_child_samples': skopt.space.Integer(10, 200), #min_data_in_leaf
        'scale_pos_weight': skopt.space.Integer(1,400),
        'subsample':skopt.space.Real(0.1,1.0),  #bagging_fraction
        'colsample_bytree':skopt.space.Real(0.1,1.0),     #feature_fraction   
        'reg_alpha': skopt.space.Integer(0,100),
        'reg_lambda': skopt.space.Integer(0,100),
         'learning_rate': skopt.space.Real(0.00001, 0.1)
       
    }
}


lcb_bo = {
            'num_leaves': (15, 1024),
                'l2_leaf_reg': [2, 18],

          #  'max_depth': (4, 20),
            'learning_rate': (0.005, 0.1),
            'bagging_fraction': (0.1, 1.0),
            'feature_fraction': (0.1, 1.0),
            'min_data_in_leaf': (6, 200),
            'bagging_freq': (0, 10),
            'reg_alpha': (0,100),
            'reg_lambda': (0,100),
        }
    
    
BO_SPACES = {
        sklearn.linear_model.Ridge.__name__: {
            'alpha': (0.001, 1000),
        },

        lgb.LGBMRegressor.__name__:  lgbm_bo,
        lgb.LGBMClassifier.__name__:  lgbm_bo,
    

        catboost.CatBoostRegressor.__name__: {
            'max_depth': [4, 12],
            'learning_rate': [0.001],

        }
    }



def subm_res(res_dic, x_text , comm = "comment",
             competition = "LANL-Earthquake-Prediction"):
    res_dic['prediction'] = blend_models(res_dic["models"],x_text)
    sub = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')
    sub['time_to_failure'] = res_dic['prediction']
    filename = 'submission_'+str(res_dic["score_cv"][0])+'.csv'
    sub.to_csv(filename)
    command = 'kaggle competitions submit '+ competition + ' -f '+filename+' -m\"' + comm + '\"'
    print(sub.head())
    print('\n\n')
    print(command, '\n\n')
    pickle_filename = res_dic["model"].__name__[:20]+"_"+str(res_dic["score_cv"][0])+".model"+".jbl"
    joblib.dump(res_dic,filename=pickle_filename,compress=jl_compress)
    return res_dic['prediction']

def list_models(dir="."):
    f_list = os.listdir(dir)             
    res = [f for f in f_list if ".model" in f]
    return res

def stack_models(file_list, X, X_test):
    for f in file_list:
        model_dict = joblib.load(f)
        X[f] = model_dict["predictions_cv"]
        X_test[f] = ['prediction']
    return X, X_test

              
class Print_Callback:
    def __init__(self):
        pass
    #    self.best_index = -1
    def __call__(self, x):
        if min(x.func_vals) == x.func_vals[-1]:
            print(datetime.datetime.now().time().strftime(format="%HH:%MM:%SS"), "new best ", x.func_vals[-1], " iter ", len(x.func_vals))

BO_RUN = {
    "lgbr": {
      "model":   
         {
        "estimator" : lgb.LGBMRegressor(n_estimators=2000, observation="mae"), 
       "search_spaces": SKOPT_BO["lgb"],
       "verbose": True,
       "n_iter":3000,
       "n_jobs":5,
       "cv":KFold(5, shuffle=True, random_state=42),
       "scoring":"neg_mean_absolute_error",
        },
        "fit_params" :{"early_stopping_rounds":200}
         }
    }


def bo(X, y, estimator = "lgbr", 
       search_spaces= {},
       verbose=True,
       n_iter=3000,
       n_jobs=5,
       cv=KFold(5, shuffle=True, random_state=42),
       scoring="neg_mean_absolute_error",
       fit_params ={},
       callbacks = [Print_Callback()],
       max_time = 7200,
       eval_set_ratio = 0.15
      ):
    if eval_set_ratio is not None and eval_set_ratio>0:
        X_train, X_early_stop, y_train, y_early_stop = train_test_split(X, y, test_size=eval_set_ratio, random_state=42)
        fit_params["eval_set"] = [(X_early_stop,y_early_stop)]
    else:
        X_train = X,
        y_train = y
    if max_time is not None and max_time>0:
        callbacks.append(skopt.callbacks.DeadlineStopper(total_time=max_time))

    if isinstance(estimator, str):
        fit_params.update(BO_RUN[estimator]["fit_params"])
        params = BO_RUN[estimator]["model"]
        if search_spaces is not None and len(search_spaces)>0: params["search_spaces"] = search_spaces
        if n_iter is not None: params["n_iter"] = n_iter
        if n_jobs is not None: params["n_jobs"] = n_jobs
        if verbose is not None: params["verbose"] = verbose
        if cv is not None: params["cv"] = cv
        if scoring is not None: params["scoring"] = scoring

        opt =  skopt.BayesSearchCV(fit_params=fit_params,**params)
    else:   
        opt = skopt.BayesSearchCV(estimator,
            search_spaces=  search_spaces,
            n_iter=n_iter,n_jobs=n_jobs,cv=cv, scoring=scoring,
            fit_params =fit_params )

    opt.fit(X=X_train,y=y_train, callback=callbacks)
    
    print(opt.best_iteration_)
    print(opt.best_score_, opt.best_params_)
    print("Byes opt res "+  str(opt.best_score_) + " " + str(opt.best_params_), file=open("output.txt", "a"))

    return opt