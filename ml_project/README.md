ML_in_production_hw1
==============================

homework 1

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

Installation:
```py
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Model Training:
LogisticRegression
```py
python -m ml_project.models.train_model configs/train_logistic.yaml
```

RandomForest
```py
python -m ml_project.models.train_model configs/train_rf.yaml
```



Model Prediction:
LogisticRegression
```py
python -m ml_project.models.predict_model configs/predict_logistic.yaml
```

RandomForest
```py
python -m ml_project.models.predict_model configs/predict_rf.yaml
```

Test:
```py
pytest tests/
```


## Самооценка
0. В описании к пул реквесту описаны основные архитектурные и тактические решения. (1 балл)
1. В пулл-реквесте проведена самооценка, распишите по каждому пункту выполнен ли критерий или нет и на сколько баллов(частично или полностью) (1 балл)
2. Выполнено EDA, закоммитьте ноутбук в папку с ноутбуками (1 балл)
3. Написана функция/класс для тренировки модели (3 балла)
4. Написана функция/класс predict(3 балла)
5. Проект имеет модульную структуру (2 балла)
6. Использованы логгеры (2 балла)
7. Написаны тесты на отдельные модули и на прогон обучения и predict (3 балла)
8. Для тестов генерируются синтетические данные (2 балла)
9. Обучение модели конфигурируется с помощью конфигов в json или yaml, закоммитьте как минимум 2 корректные конфигурации, с помощью которых можно обучить модель (разные модели, стратегии split, preprocessing) (3 балла)
10. Используются датаклассы для сущностей из конфига, а не голые dict (2 балла)
11. Напишите кастомный трансформер и протестируйте его (3 балла)
12. В проекте зафиксированы все зависимости (1 балл)