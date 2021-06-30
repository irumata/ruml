# ruml
rumata ml utils

***Installation***

1. Copy files to your computer and go to folder
2. Run in terminal
python3 setup.py build
python3 setup.py install

3. In python 3 you able to write "import ruml"

to run method

from ruml import lr_clust

**Fit regressor**
fit
Packeg provide to types of method: lr_on_leaf.
lr_on_leaf class has predict and fit method.
lr_on_leaf implements described method, and has the next paramatres
X - train set 
y - target value
Optional you also can provide parameters for fitting on cluster and regression steps, such as:
lr_columns - which columns to use for the final step (after clustering)
cls_columns - which columns to use on clustering step
n_clusters - number of clusters 
verbose - verbose level
eval_set - set for hyperparameters tuning
n_estimators - estimators number for GBM 
max_depth - for GBM

example:
lr_cl = lr_clust.lr_on_leaf()
lr_cl.fit(X_train,y_train,lr_columns=tr_cols_reg,cls_columns=tr_cols_clust2,n_clusters=3,verbose=False,
         eval_set=(X_test,y_test), n_estimators=40, max_depth=4)

**prediction**
to predict use predict method
lr_cl.predict(X_test)

For testing reasons it's able to predict only by best estimator
lr_cl.predict_best(X_test,y_test)

**Other function**
You can use provide lr_as_feature class, which represents similar method, but instead of regression over clustering uses cluster regressions results as feautures for clusterization.

On utils backege bayes optimization and several other techniques also provided.




