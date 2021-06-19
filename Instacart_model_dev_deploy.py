# read the data into dataframe
import pandas as pd
bucket='imba'
data_key = 'output_small/data_small.csv'
data_location = 's3://{}/{}'.format(bucket, data_key)
data = pd.read_csv(data_location)

# load order product train dataframe
order_product_train_key = 'data/order_products/order_products__train.csv.gz'
order_product_train_location = 's3://{}/{}'.format(bucket, order_product_train_key)

order_product_train = pd.read_csv(order_product_train_location)

# load orders dataframe
orders_key = 'data/orders/orders.csv'
orders_location = 's3://{}/{}'.format(bucket, orders_key)

orders = pd.read_csv(orders_location)
# only select train and test orders
#orders = orders[orders.eval_set != 'prior'][['user_id', 'order_id','eval_set']]

# attach user_id to order_product_train
order_product_train = order_product_train.merge(orders[['user_id', 'order_id']])

# attach eval_set to data
#data = data.merge(orders[orders.eval_set != 'prior'][['user_id', 'order_id','eval_set']])
data = data.merge(orders[orders.eval_set != 'prior'][['user_id','eval_set']])

# attach target variable: reordered
data = data.merge(order_product_train[['user_id', 'product_id', 'reordered']], how = 'left')

data['prod_reorder_probability'] = data.prod_second_orders / data.prod_first_orders
data['prod_reorder_times'] = 1 + data.prod_reorders / data.prod_first_orders
data['prod_reorder_ratio'] = data.prod_reorders / data.prod_orders
data.drop(['prod_reorders', 'prod_first_orders', 'prod_second_orders'], axis=1, inplace=True)
data['user_average_basket'] = data.user_total_products / data.user_orders
data['up_order_rate'] = data.up_orders / data.user_orders
data['up_orders_since_last_order'] = data.user_orders - data.up_last_order
data['up_order_rate_since_first_order'] = data.up_orders / (data.user_orders - data.up_first_order + 1)

# split into training and test set, test set does not have target variable
train = data[data.eval_set == 'train'].copy()
test = data[data.eval_set == 'test'].copy()

# id field won't be used in model, thus make a backup of them and remove from dataframe
#test_id = test[['product_id','user_id', 'order_id', 'eval_set']]
test_id = test[['product_id','user_id', 'eval_set']]
#test.drop(['product_id','user_id', 'order_id', 'eval_set', 'reordered'], axis=1, inplace=True)
test.drop(['product_id','user_id', 'eval_set', 'reordered'], axis=1, inplace=True)

# convert target variable to 1/0 for training dataframe
train['reordered'] = train['reordered'].fillna(0)
train['reordered'] = train.reordered.astype(int)

# drop id columns as they won't be used in model
#train.drop(['eval_set', 'user_id', 'product_id', 'order_id'], axis=1, inplace=True)
train.drop(['eval_set', 'user_id', 'product_id'], axis=1, inplace=True)

# this is the target variable dataframe
train_y = train[['reordered']]
# this is the dataframe without target variable
train_X = train.drop(['reordered'], axis = 1)

## Step2 classification
import pandas as pd

val_X = train_X[:20000]
train_X = train_X[20000:]

val_y = train_y[:20000]
train_y = train_y[20000:]

#test_y = pd.DataFrame(test_y)
#test_X = pd.DataFrame(test_X)

import os
data_dir = 'data/xgboost'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# First, save the test data to test.csv in the data_dir directory without label.
pd.DataFrame(test).to_csv(os.path.join(data_dir, 'test.csv'), header=False, index=False)

# Then save the training and validation set into local disk as csv files
pd.concat([val_y, val_X], axis=1).to_csv(os.path.join(data_dir, 'validation.csv'), header=False, index=False)
pd.concat([train_y, train_X], axis=1).to_csv(os.path.join(data_dir, 'train.csv'), header=False, index=False)

# To save a bit of memory set text_X, train_X, val_X, train_y and val_y to None.

train_X = val_X = train_y = val_y = None

import sagemaker

session = sagemaker.Session() # Store the current SageMaker session

# S3 prefix (which folder will use)
prefix = 'imba-xgboost'

test_location = session.upload_data(os.path.join(data_dir, 'test.csv'), key_prefix=prefix)
val_location = session.upload_data(os.path.join(data_dir, 'validation.csv'), key_prefix=prefix)
train_location = session.upload_data(os.path.join(data_dir, 'train.csv'), key_prefix=prefix)

from sagemaker import get_execution_role

# current execution role is require when creating the model as the training
# and inference code will need to access the model artifacts.
role = get_execution_role()

from sagemaker.amazon.amazon_estimator import get_image_uri

container = get_image_uri(session.boto_region_name, 'xgboost', '0.90-1')

xgb = sagemaker.estimator.Estimator(container, # The location of the container we wish to use
                                    role,                                    # What is our current IAM Role
                                    train_instance_count=1,                  # How many compute instances
                                    train_instance_type='ml.m4.xlarge',      # What kind of compute instances
                                    output_path='s3://{}/{}/output'.format(session.default_bucket(), prefix),
                                    sagemaker_session=session)

#       Set the XGBoost hyperparameters in the xgb object.  binary
#       label should be using the 'binary:logistic' objective.

# Solution:
xgb.set_hyperparameters(max_depth=5,
                        eta=0.1,
                        gamma=4,
                        min_child_weight=6,
                        subsample=0.8,
                        silent=0,
                        objective='binary:logistic',
                        early_stopping_rounds=10,
                        num_round=500)

from sagemaker.tuner import IntegerParameter, ContinuousParameter, HyperparameterTuner


# create the tuner object:

xgb_hyperparameter_tuner = HyperparameterTuner(estimator = xgb, # The estimator object to use as the basis for the training jobs.
                                               objective_metric_name = 'validation:rmse', # The metric used to compare trained models.
                                               objective_type = 'Minimize', # Whether we wish to minimize or maximize the metric.
                                               max_jobs = 4, # The total number of models to train
                                               max_parallel_jobs = 3, # The number of models to train in parallel
                                               hyperparameter_ranges = {
                                                    'max_depth': IntegerParameter(3, 12),
                                                    'eta'      : ContinuousParameter(0.05, 0.5),
                                                    'min_child_weight': IntegerParameter(2, 8),
                                                    'subsample': ContinuousParameter(0.5, 0.9),
                                                    'gamma': ContinuousParameter(0, 10),
                                               })
s3_input_train = sagemaker.TrainingInput(s3_data=train_location, content_type='csv')
s3_input_validation = sagemaker.TrainingInput(s3_data=val_location, content_type='csv')
xgb_hyperparameter_tuner.fit({'train': s3_input_train, 'validation': s3_input_validation})

# attach the model:

xgb_attached = sagemaker.estimator.Estimator.attach(xgb_hyperparameter_tuner.best_training_job())

xgb_transformer = xgb_attached.transformer(instance_count = 1, instance_type = 'ml.m4.xlarge')

xgb_transformer.transform(test_location, content_type='text/csv', split_type='Line')

!aws s3 cp --recursive $xgb_transformer.output_path $data_dir

import boto3

runtime = boto3.Session().client('sagemaker-runtime')

xgb_predictor = xgb_attached.deploy(initial_instance_count = 1, instance_type = 'ml.m4.xlarge')

xgb_predictor.endpoint

response = runtime.invoke_endpoint(EndpointName = xgb_predictor.endpoint, # The name of the endpoint we created
                                       ContentType = 'text/csv',                     # The data format that is expected
                                       Body = '1,6.67578125,3418,209,0.595703125,514,10.0,11,57,11,1599,0.1498791297340854,1.2884770346494763,0.22388993120700437,9.017543859649123,0.017543859649122806,46,0.02127659574468085')
response['Body'].read().decode('utf-8')

predictions = pd.read_csv(os.path.join(data_dir, 'test.csv.out'), header=None)
predictions = [round(num) for num in predictions.squeeze().values]