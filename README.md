# Ведение DS-проектов"

## Сохранение результатов эксперимента (пример)

### **Данные**,

Португальские вина «Vinho Verde»,

* 12 параметров,
* до 15% пропусков,
* есть категориальные признаки

### **Обработка данных**

1. Заполняем пропуски средними значениями
2. Переводим категориальные признаки в числовые
3. Нормализуем данные 

### **Модели**,

Для решения задачи используем **регрессию**
Используем ML модели. Точность указана на тестовой выборке:

1. LinearRegression 87,67%
2. LinearSVR 87,69%
3. RandomForestRegressor 87,32%
4. Stacking (ансамбль) 98,56%

### **Результат**
В данном исследовании лучший результат показала ансамблевая модель (стекинг 3х моделей)
