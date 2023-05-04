from deep_perm_imp import ImagePermI
import skexplain
from skexplain.common.importance_utils import to_skexplain_importance

#loadd in the models

X = ds_train.features.astype('float16').values[:,:,:,:]
y = ds_train.label_1d_class.astype('float16').values

# We are only using 75% of the training images to determine importance. 
# It is not neccesary to use the full training dataset since 
# we are generating bulk statistics. 
subsample_size=0.75

ip = ImagePermI(X, model_class, y, subsample_size=subsample_size)
ip.single_pass(direction='backward', metric='auc')

# Since we are ranking from high to low, we need to convert the 
# permuted scores into importance scores
imp_scores = ip.start_score - ip.scores

# We set the method as 'backward_singlepass'. This is an expected name in skexplain and will
# help with the plotting. 

# to_skexplain_importance is expecting the data in 
# following shape (n_features, n_permutes). For this notebook, 
# we only performed one permutation so we need to convert imp scores to (n_features, 1)
results = to_skexplain_importance(
    importances=np.array([imp_scores]).T, 
    estimator_name='CNN', 
    feature_names=ds_train.n_channel.values,
    method='backward_singlepass', normalize=False
)

# Since we are using backward permutation importance we will want to include the
# original score which is used for comparison to measure importance. 
results['original_score__CNN'] = (('n_boot'), [ip.start_score])

# Lastly, it is easier to interpret the feature importances if we convert back to the permuted scores.
results['backward_singlepass_scores__CNN'] =  ip.start_score - results['backward_singlepass_scores__CNN']