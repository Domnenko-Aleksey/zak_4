{"nbformat":4,"nbformat_minor":0,"metadata":{"colab":{"name":"README.md","provenance":[],"collapsed_sections":[],"authorship_tag":"ABX9TyP/2vwOP8ejnPo/nLrH2ZpX"},"kernelspec":{"name":"python3","display_name":"Python 3"},"language_info":{"name":"python"}},"cells":[{"cell_type":"markdown","source":["# Ведение DS-проектов\n","## Сохранение результатов эксперимента (пример)\n","\n","\n","### **Данные**\n","Португальские вина «Vinho Verde»,\n","\n","* 12 параметров\n","\n","* до 15% пропусков\n","\n","* есть категориальные\n","признаки\n","\n","### **Обработка данных**\n","\n","1. Заполняем пропуски\n","средними значениями\n","\n","2. Переводим\n","категориальные\n","признаки в числовые\n","\n","3. Нормализуем данные\n","\n","### **Модели**\n","Для решения задачи используем **регрессию**\n","\n","Используем ML модели. Точность указана на тестовой выборке:\n","\n","1. LinearRegression 87,67%\n","\n","2. LinearSVR 87,69%\n","\n","3. RandomForestRegressor 87,32%\n","\n","4. Stacking (ансамбль) 98,56%\n","\n","### **Результат**\n","\n","В данном исследовании лучший результат показала ансамблевая модель (стекинг 3х моделей)"],"metadata":{"id":"NIf7pCqH-LoR"}}]}