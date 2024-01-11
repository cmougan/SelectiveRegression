import os.path

import openml
#openml.config.apikey = 'FILL_IN_OPENML_API_KEY'  # set the OpenML Api Key
SUITE_ID = 336 # Regression on numerical features
#SUITE_ID = 337 # Classification on numerical features
#SUITE_ID = 335 # Regression on numerical and categorical features
#SUITE_ID = 334 # Classification on numerical and categorical features
benchmark_suite = openml.study.get_suite(SUITE_ID)  # obtain the benchmark suite
for task_id in benchmark_suite.tasks:  # iterate over all tasks

    task = openml.tasks.get_task(task_id)  # download the OpenML task
    dataset = task.get_dataset()
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )
    df = X.copy()
    df["TARGET"] = y
    data_folder = "data/grin_benchmark/{}".format(dataset.name)
    if os.path.exists(data_folder) is False:
        os.mkdir(data_folder)
    if os.path.exists("{}/{}.csv".format(data_folder, dataset.name)) is False:
        df.to_csv("{}/{}.csv".format(data_folder, dataset.name), index=False)

SUITE_ID = 335 # Regression on numerical and categorical features
benchmark_suite = openml.study.get_suite(SUITE_ID)
for task_id in benchmark_suite.tasks:  # iterate over all tasks
    task = openml.tasks.get_task(task_id)  # download the OpenML task
    dataset = task.get_dataset()
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )
    df = X.copy()
    df["TARGET"] = y
    data_folder = "data/grin_benchmark/{}".format(dataset.name)
    if os.path.exists(data_folder) is False:
        os.mkdir(data_folder)
    if os.path.exists("{}/{}.csv".format(data_folder, dataset.name)) is False:
        df.to_csv("{}/{}.csv".format(data_folder, dataset.name), index=False)