# AestheticAssessment
Для начала надо как указано в instructions.txt запустить все что касается данных (в секции datasets) в том порядке, в котором указано (сверху вниз). Если что связь между файлами указана, по логике порядок установить не сложно (смотреть что что делает и какие файлы использует/создает).

для моделей веса взяты из этих репозиториев (соответствующие подпапки это просто скопированные эти, так что просто поставить по соответствущим путям.)
https://github.com/alanspike/personalizedImageAesthetics/tree/master
https://github.com/woshidandan/TANet

Затем чтобы протестить модель TANet можно запускать написанный руками файлы my_val.py в TANet/code/TAD66k и TANet/code/AVA:
    Использует full_set для предикта. Результат записывает в my_res.csv.
    Команда: python my_val.py
    Чтобы посмотреть графики корреляции смотреть instructions.txt файл TANet_check в разделе code/stage1

Также чтобы протестить personalizedImageAesthetics можно запустить следующие команды в personalizedImageAesthetics, чтобы получить предикт на Рандомных 10k и наших размеченных данных:
    Наши данные:
        Использует image_list.txt (формируется из full_set.csv, в файле beauty_predict/datasets/full_set_to_personalizedImageAesthetics.py)
        На выход выдает res.json. Для дальнейшей работы с результатами надо переименовать в full_set_res.json . 
        Команда: python test.py --filename image_list.txt
        Чтобы посмотреть графики корреляции смотреть instructions.txt файл personalizedImageAesthetics_check в разделе code/stage1
        
    10k:
        Использует big_set.txt (формируется из big_set.csv, смотреть beauty_predict/datasets)
        На выход выдает res.json. Для дальнейшей работы с результатами надо переименовать в big_set_res.json . (чтобы отобразить хорошие из них нужно запустить beauty_predict/code/stage2/save_good_predicts.py)
        Команда: python test.py --filename big_set.txt
        Чтобы посмотреть графики корреляции смотреть instructions.txt файл personalizedImageAesthetics_check в разделе code/stage1
        
Гипотезы 1) о связи таргета с расстоянием до центроиды, 2) о связи таргета и косинусным расстоянием между эмбеддингами картинки и текста смотреть в instructions.txt в разделе code/stage2
