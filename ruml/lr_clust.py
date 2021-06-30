from sklearn.linear_model import LinearRegression, Ridge
import pandas as pd
from sklearn.metrics import mean_absolute_error, median_absolute_error
import numpy as np
import copy
from catboost import CatBoostClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import inspect
from .utils import *

#Return coefficient of linear regression
def dict_coefs_columns(lr, columns):
    return dict(zip(columns,lr.coef_))   

def get_cluster_model (X_train, y_train, X_test=None, y_test=None,check_func=None, 
        fit_params={}, model_params={}, model = Ridge
      ):
    model_inst = model(**model_params)
    model_inst.fit(X_train,y_train, **fit_params)
    return model_inst

CLUSTER_PREF  = "cl_"
def lr_k_means(X_train,y_train,n_clusters = 5,
               max_iter = 20 , init_seed=None, 
               min_cluster_elem=10, print_level = 2, 
               fit_params={}, model_params={}, model = Ridge,
               metric = mean_absolute_error
               ):

    columns = X_train.columns
    
    cluster_list = [CLUSTER_PREF+str(i) for i in range(n_clusters)]
    rand_cluster = pd.Series(data=np.random.randint(n_clusters, size=X_train.shape[0]), index = X_train.index ).astype(str)

    clusters =  pd.Series(data=CLUSTER_PREF +rand_cluster ,
                          index = X_train.index,
                         dtype = str)
    if print_level> 0: print("first 2 clusters",clusters.value_counts()[:2])
    n_iter = 0
    first = True
    clusters_prev=clusters
    preds_best = pd.Series([0]*len(X_train), index = X_train.index)

    while first or ( (n_iter<max_iter) and (clusters_prev !=clusters).sum()>0):
        clusters_prev = clusters
        errors = {}
        cluster_models = {}
        for cluster in cluster_list:
            mask =  (clusters==cluster)
            elem_in_cluster = mask.sum()
            if elem_in_cluster<min_cluster_elem:
                X_train[cluster] =np.finfo("float").max 
                continue
            X_train_cl = X_train[mask][columns]
            y_train_cl =  y_train[mask]
            if first and init_seed is not None:
                 cl_model = get_cluster_model(X_train_cl,
                                         y_train_cl, 
                 fit_params=fit_params, model_params=model_params, model = model
                                          
                                          )
            else:
                 cl_model = get_cluster_model(X_train_cl, 
                                         y_train_cl,
                 fit_params=fit_params, model_params=model_params, model = model)
            cluster_models[cluster] = cl_model
            preds = conv_pred(cl_model.predict(X_train_cl))

            errors[cluster] = metric(y_train_cl, preds)
            preds_best[mask] = preds
            
            preds = conv_pred(cl_model.predict(X_train[columns]))
            
            X_train[cluster] = preds
            X_train[cluster] = (X_train[cluster] - y_train).abs()
            
        first=False
        clusters= X_train[cluster_list].idxmin(axis=1)
        X_train["clusters"] = clusters
        if print_level>1 and n_iter%5 == 0:
            metric_best = metric(preds_best, y_train)

            print("clusters iter ",n_iter,clusters.value_counts(), errors, "best train", metric_best)
        n_iter+=1

        X_train["clusters"] = clusters
    if print_level>0: 
        metric_best = metric(preds_best, y_train)

        print("result ",n_iter,clusters.value_counts(),  errors, "best train all ", metric_best)
    X_train.drop(labels=cluster_list+["clusters"], axis=1,inplace=True)
    str_to_int = {t:i  for i,t in enumerate(cluster_list)}
    clusters = clusters.map(str_to_int)
    cluster_models = { str_to_int[k]:v for k,v in cluster_models.items()}
    return (clusters, cluster_models, errors)
    
    

from sklearn.datasets import make_multilabel_classification
from sklearn.tree import DecisionTreeClassifier
class Cluster_Cls:
    
    def __init__(self,model=lgb.LGBMClassifier, cluster_models = None, **model_params):
        if "cls_model" in model_params.keys():
            self.model = model_params["cls_model"](**{k:v for k,v in model_params if k != "cls_model" })
        else:
            if issubclass(model,lgb.LGBMClassifier) and "objective" not in model_params.keys():
                
                model_params["objective"] = "multiclass"
            self.model = model(**model_params)
        self.cluster_models_ = cluster_models
            
    def set_cluster_models(self,cluster_models):
        self.cluster_models_ = cluster_models
        
        
    def fit (self, X_train,clusters, **fit_params):
        self.model.fit(X_train,clusters, **fit_params)
        return self
    
    def predict(self, X, cluster_models=None):
        if cluster_models is not None:
             self.cluster_models_ = cluster_models
        cls_cols = X.columns
        preds = self.model.predict(X)
        if (isinstance(preds,np.ndarray) and isinstance(preds[0],np.ndarray)) or (
                isinstance(preds,pd.Series) and isinstance(preds.iloc[0],np.ndarray)):
            preds = preds[:,0]
        X["cl"] = preds
        res = X.apply(axis=1, func = lambda x:self.cluster_models_[x["cl"]].predict([x[cls_cols]] ) )
      #  print( "X",X[:5], preds[:5], res[:5], "value_counts",pd.Series(preds).value_counts())
        X.drop(["cl"], axis=1,inplace=True)

        if isinstance(res.iloc[0], np.ndarray):
            res = res.apply(lambda x : x[0])
        return res
    
    def predict_best(self,X,y):
        temp_cols = list()
        temp_colas = list()
        cols = X.columns
       # print("y", y[:5] , X[:5])
        for i,model in self.cluster_models_.items():
            col = "c"+str(i)
            cola =  "ca"+str(i)
            X[col] = model.predict(X[cols])
            X[cola]= (X[col] - y).abs()
            temp_cols.append(col)
            temp_cols.append(cola)
            temp_colas.append(cola)        
        X["opti"] = X[temp_colas].idxmin(axis=1)
        temp_cols.append("opti")
        res = X.apply(lambda x: x["c"+x["opti"][-1]], axis=1)

        X.drop(temp_cols, axis=1,inplace=True)
        
        return res
        



    
class lr_on_leaf:
    def __init__(self, **params):
        self.cls = Cluster_Cls(**params)
        self.__name__="lr_on_l_"+type(self.cls.model).__name__


    
    def fit(self,X,y,eval_set=None, **fit_params):
        
        lr_k_means_args = {k:v for k,v in fit_params.items() if k in inspect.getfullargspec(lr_k_means).args}
        
        clusters, cluster_models,errors = lr_k_means(X,y,**lr_k_means_args)
        self.cls.set_cluster_models(cluster_models)
        
        cls_fit_args = {k:v for k,v in fit_params.items() if k in inspect.getfullargspec( self.cls.model.fit).args}
        if eval_set is not None:
            if isinstance(eval_set[0],tuple):
                eval_set = eval_set[0]
            X_val = eval_set[0](".")
            y_val = eval_set[1]
            cls_fit_args["eval_set"] = [(X_val,get_classes(X_val,y_val,cluster_models))]
            
        

    
        self.cls.fit(X,clusters,**cls_fit_args)
        
    def predict(self,X):
        return self.cls.predict(X)

            
    def predict_best(self,X,y):
        return self.cls.predict_best(X,y)
    
    def get_cluster_models(self):
         return self.cls.cluster_models_

        
class lr_as_feature:
    def __init__(self, **model_params):
        if "cls_model" in model_params.keys():
            self.model = model_params["cls_model"](**{k:v for k,v in model_params if k != "cls_model" })
        else:
#             if issubclass(model,lgb.LGBMRgressor) and "objective" not in model_params.keys():
                
#                 model_params["objective"] = "multiclass"
            self.model = lgb.LGBMRegressor(**model_params)
        self.cluster_models_ = None
        self.__name__="lr_as_f_"+type(self.model).__name__
    
    def fit(self,X,y,eval_set=None, **fit_params):
        
        lr_k_means_args = {k:v for k,v in fit_params.items() if k in inspect.getfullargspec(lr_k_means).args}
        self.lr_cols_ = X.columns
        
        clusters, cluster_models,errors = lr_k_means(X,y,**lr_k_means_args)
        self.cluster_models_ = cluster_models
        cls_fit_args = {k:v for k,v in fit_params.items() if k in inspect.getfullargspec(self.model.fit).args}
        temp_cols = list()
        if eval_set is not None:
            if isinstance(eval_set[0],tuple):
                eval_set = eval_set[0]
            X_eval =  eval_set[0]
        for i, model_cluster in cluster_models.items():
            col = "cl_"+str(i)
            X[col] = conv_pred(model_cluster.predict(X[self.lr_cols_ ]))
            if eval_set is not None:
                X_eval[col] = conv_pred(model_cluster.predict(X_eval[self.lr_cols_ ]))

            temp_cols.append(col)

        
        cls_fit_args = {k:v for k,v in fit_params.items() if k in inspect.getfullargspec( self.model.fit).args}
        if eval_set is not None:
            cls_fit_args["eval_set"] = [( X_eval, eval_set[1])]
       # print(X.columns, len(X.columns), X.shape, cls_fit_args )    
        self.model.fit(X,y,**cls_fit_args)
        X.drop(temp_cols, axis=1, inplace=True)
        
    def predict(self,X):
        temp_cols = list()
        for i, model_cluster in self.cluster_models_.items():
            col = "cl_"+str(i)
            X[col] = conv_pred(model_cluster.predict(X[self.lr_cols_ ]))
            temp_cols.append(col)
            
        res = self.model.predict(X)
        X.drop(temp_cols, axis=1, inplace=True)
        return res

    def get_cluster_models(self):

        
        return self.cluster_models_
                
        

def get_classes(X,y,cluster_models):
    temp_cols = list()
    cls_cols = X.columns
    for i,model in cluster_models.items():
        col = "cl_col_" +str(i)
        X[col] = conv_pred(model.predict(X[cls_cols]))
        X[col]= (X[col] - y).abs()
        temp_cols.append(col)
    res = X[temp_cols].idxmin(axis=1)
    res = res.apply(lambda x: int(x[7:]) )
    X.drop(temp_cols,axis=1,inplace=True)
    return res
        
    
    
    
        
    
    
    
 
