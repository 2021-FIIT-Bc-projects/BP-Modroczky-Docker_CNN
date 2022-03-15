# Bakalárska práca

## Zapuzdrené prostredie pre spracovanie obrázkov na rozpoznávanie húb

Autor: Viktor Modroczký

Vedúci práce: Ing. Giang Nguyen Thu, PhD.

### Zadanie

Zapuzdrené prostredie (angl. containerization) je dnes kritickým prvkom pre vývoj softvéru. Jeho použitie
umožňuje vybudovať inteligentné aplikácie od prototypu k nasadeniu pohodlne a časovo efektívne.
Momentálne, najpopulárnejší typ inteligentnej aplikácie je spracovanie obrázkov pomocou konvolučných
neurónových sietí (CNN). Analyzujte súčasný stav kontajnerizácie softvérov a spracovania obrázkov pomocou
CNN. Trénujte inteligentný model na rozpoznávanie húb (jedlé a jedovaté) na základe fotografií z Google
Images alebo z iných verejne dostupných zdrojov. Ako praktickú ukážku vybudovania zapuzdreného
prostredia, vytvorte docker kontajner s aplikáciou pre dátovú vedu. Z vytrénovaného modelu realizujte flask
aplikáciu na vizualizáciu. Vyhodnoťte výsledný softvérový produkt vrátane kvality vytvoreného modelu.

### Skripty

Pred spúšťaním skriptov treba nainštalovať potrebné knižnice.

```text
python -m pip install -r requirements.txt
```

#### Skript pre získanie obrázkov do trénovacieho a testovacieho datasetu

```text
python obtain.py <path/to/json/file>
```

Príklad metadát v json súbore pre skript obtain.py:

```json
{
    "tsv_path": "path/to/mushroom/observer/tsv/file",
    "dl_path": "path/to/download/folder",
    "authors_path": "path/to/save/author/names",
    "queries": [
        "amanita",
        "boletus",
        "cantharellus",
        "morchella",
        "macrolepiota",
        "craterellus",
        "pleurotus",
        "psilocybe"
    ],
    "limit": 3000
}
```

`tsv_path` je umiestnenie súboru [tsv](https://drive.google.com/file/d/1fPXJtJpqiQEQb1ezINdFK-Jhee84DvMA/view?usp=sharing), ktorý obsahuje zoznam obrázkov húb z Mushroom Observer.

`dl_path` je priečinok, do ktorého sa majú obrázky sťahovať.

`authors_path` je priečinok, do ktorého sa má uložiť textový súbor s menami autorov obrázkov.

`queries` je zoznam názvov húb, ktoré sa majú stiahnuť.

`limit` je maximálny počet obrázkov, ktorý sa má stiahnuť pre jeden typ huby.

#### Skript pre rozšírenie trénovacieho datasetu

```text
python augment.py <path/to/json/file>
```

Príklad metadát v json súbore pre skript augment.py:

```json
{
    "img_size": 299,
    "classes": {
        "train": {
            "amanita": 58,
            "boletus": 48,
            "cantharellus": 49,
            "morchella": 46,
            "macrolepiota": 97,
            "craterellus": 55,
            "pleurotus": 34,
            "psilocybe": 80
        },
        "test": {
            "amanita": 6,
            "boletus": 8,
            "cantharellus": 7,
            "morchella": 7,
            "macrolepiota": 8,
            "craterellus": 7,
            "pleurotus": 6,
            "psilocybe": 7
        }
  },
  "data_path": "../../data/dataset",
  "augmented_data_path": "../../data/dataset_augmented_inception"
}
```

`img_size` je veľkosť výstupných obrázkov.

`classes` je zoznam tried v trénovacom a testovacom priečinku pre klasifikáciu spolu s číslom, ktoré hovorí, koľkokrát sa má každý obrázok v danej triede rozšíriť. Ak sa uvedie iba zoznam pre trénovací alebo testovací dataset, tak rozširovanie prebehne len pre uvedený dataset.

`data_path` je umiestnenie trénovacieho ($data_path/train) a testovacieho datasetu ($data_path/test).

`augmented_data_path` je nové umiestnenie rozšíreného testovacieho datasetu ($augmented_data_path/test) a trénovacieho datasetu ($augmented_data_path/train).

### Experiment 1 - VGG16

Python Notebook [vgg16.ipynb](src/vgg16.ipynb) bol spúšťaný na platforme Kaggle a je získaný z odkazu <https://www.kaggle.com/viktormodroczky/vgg16-for-fungi-classification/notebook>.

<details>
<summary><b>Vybudovaný model VGG16 pomocou Keras</b></summary>

![VGG16](plots/vgg16_plot.png)

</details>

### Experiment 2
