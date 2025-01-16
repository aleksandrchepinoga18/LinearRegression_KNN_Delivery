# LinearRegression_KNN_Delivery
Прогнозирование времени доставки заказов

# Проект: Прогнозирование времени доставки заказов

## Описание проекта

Данный проект направлен на решение задачи прогнозирования времени доставки заказов для служб доставки еды. На основе предоставленных данных о заказах, таких как координаты ресторана и клиента, возраст и рейтинг доставщика, тип заказанных продуктов и транспортное средство доставщика, строится модель, которая предсказывает время доставки в минутах. Это позволяет службам доставки лучше планировать свои ресурсы, оптимизировать работу курьеров и улучшать качество обслуживания клиентов.

## Используемые библиотеки и методы

- **Библиотеки**:
  - `numpy` — для работы с массивами и математическими операциями.
  - `pandas` — для обработки и анализа данных.
  - `seaborn` и `matplotlib` — для визуализации данных.
  - `scikit-learn` — для построения моделей машинного обучения, включая линейную регрессию и метод k-ближайших соседей (k-NN), а также для предобработки данных (масштабирование, разделение на выборки).

- **Методы**:
  - Линейная регрессия — для предсказания времени доставки.
  - Метод k-ближайших соседей (k-NN) — для классификации заказов на быстрые и долгие.
  - Масштабирование данных с помощью `MinMaxScaler` — для улучшения качества моделей.
  - Метрики оценки: `mean_absolute_error` для регрессии и `accuracy_score` для классификации.

## Решаемые задачи

1. **Предсказание времени доставки**:
   - На основе данных о заказе (возраст и рейтинг доставщика, расстояние до клиента) строится модель линейной регрессии, которая предсказывает время доставки в минутах.

2. **Классификация заказов**:
   - Заказы классифицируются на быстрые (доставка занимает не более 30 минут) и долгие. Для этого используется метод k-ближайших соседей (k-NN).

3. **Оптимизация работы службы доставки**:
   - Прогнозирование времени доставки позволяет службам доставки лучше планировать свои ресурсы, подбирать курьеров для заказов и точнее уведомлять клиентов о времени доставки.

## Результаты

1. **Линейная регрессия**:
   - Средняя абсолютная ошибка (MAE) на тестовых данных составила **6.33 минуты** без масштабирования признаков и **6.26 минуты** с масштабированием.

2. **Классификация с использованием k-NN**:
   - Точность модели (accuracy) на тестовых данных составила **73%** без масштабирования и **80.56%** с масштабированием.
   - Отчет о классификации показывает, что модель лучше справляется с предсказанием быстрых доставок (precision для класса 0 — 83%), чем долгих (precision для класса 1 — 73%).

3. **Визуализация данных**:
   - Для наглядности данные были визуализированы с использованием метода PCA для уменьшения размерности.

## Как использовать проект

1. Установите необходимые библиотеки:
   ```bash
   pip install numpy pandas seaborn matplotlib scikit-learn

# Пример кода:
```
#Загрузите данные
```import pandas as pd
df = pd.read_csv("https://dc-edu.itmo.ru/assets/courseware/v1/344bff205404e315b59430dffaedf04b/asset-v1:ITMO+DS+2024+type@asset+block/deliverytime.csv")

# Выполните предобработку данных:
df = df.drop(['ID', 'Delivery_person_ID'], axis=1)
df = df[df['distance_meters'] <= 100000]

# Обучите модель линейной регрессии:
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

y = df["Time_taken_min"]
X = df.drop(columns=["Time_taken_min", "order_Drinks", "order_Meal", "order_Snack",
                     "vehicle_electric_scooter", "vehicle_motorcycle", "vehicle_scooter"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
y_pred = reg_model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))

# Обучите модель k-NN для классификации:
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df['is_long_Delivery'] = df['Time_taken_min'] > 30
df['is_long_Delivery'] = df['is_long_Delivery'].astype(int)
df = df.drop(["Time_taken_min"], axis=1)

y = df["is_long_Delivery"]
X = df.drop(["is_long_Delivery"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25, stratify=y)

knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Визуализируйте результаты:
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_test_2d = pca.fit_transform(X_test)

plt.figure(figsize=(12, 6))
sns.scatterplot(x=X_test_2d[:, 0], y=X_test_2d[:, 1], hue=y_pred, palette="deep")
plt.title("KNN Predictions on Test Data")
plt.show()
