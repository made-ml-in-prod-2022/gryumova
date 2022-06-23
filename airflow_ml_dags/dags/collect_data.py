from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.docker.operators.docker import DockerOperator

from utils import load_args


DAG_NAME = 'collect_data'

default_args, dag_args, tasks_args = load_args(__name__, DAG_NAME)

with DAG(DAG_NAME, default_args=default_args, **dag_args) as dag:
    start_task = DummyOperator(task_id='begin-collect-data')

    data_merge_args = tasks_args['data_merge']
    input_paths = [f'--input-paths \"{p}\" '
                   for p in data_merge_args["input_paths"]]
    output_path = f'--output-path \"{data_merge_args["output_path"]}\"'
    data_merge_command = 'python data_merge.py ' + \
                         ''.join(input_paths) + output_path
    data_merge = DockerOperator(
        task_id='merge-data',
        image='airflow-data-download',
        command=data_merge_command,
        **tasks_args['default_args'],
    )

    data_clean_args = tasks_args['data_clean']
    input_paths = [f'--input-paths \"{p}\" '
                   for p in data_clean_args["input_paths"]]
    data_clean_command = 'python data_clean.py ' + \
                         ''.join(input_paths)
    data_clean = DockerOperator(
        task_id='clean-data',
        image='airflow-data-utils',
        command=data_clean_command,
        **tasks_args['default_args'],
    )

    sources = ['hive', 'clickhouse', 'mongo']
    data_tasks = []
    for source in sources:
        source_args = tasks_args[f'data_download_{source}']
        source_command = 'python data_download.py ' + \
                         f'--name {source_args["name"]} ' \
                         f'--output-path \'{source_args["output_path"]}\' ' \
                         f'--seed {source_args["seed"]}'
        data_download_source = DockerOperator(
            task_id=f"data-download-{source}",
            image="airflow-data-download",
            command=source_command,
            **tasks_args['default_args'],
        )

        data_tasks.append(data_download_source)

    start_task >> data_tasks >> data_merge >> data_clean
