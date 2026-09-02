# Sistema didattico per la stima del rischio di diabete

Progetto finale per il corso **Laboratorio di Intelligenza Artificiale Applicata**.

Il progetto analizza il dataset **PIMA Indians Diabetes** e sviluppa un sistema di classificazione binaria capace di stimare la probabilità che un'osservazione appartenga alla classe associata al diabete (`0` oppure `1`). Il sistema comprende l'analisi esplorativa dei dati, il confronto tra due modelli di machine learning, una web application Flask, un'API JSON, i test automatici e la configurazione Docker.

> **Avvertenza:** il progetto ha esclusivamente finalità didattiche. Il risultato non è una diagnosi, non è un dispositivo medico e non deve sostituire la valutazione di un professionista sanitario.

## Obiettivi e risultati

L'obiettivo è costruire una procedura completa e riproducibile, dalla lettura dei dati alla predizione tramite interfaccia web.

Nel progetto ho confrontato:

- **Logistic Regression**, scelta come modello finale perché è interpretabile e ha ottenuto le prestazioni medie più equilibrate nella cross-validation;
- **K-Nearest Neighbors (KNN)**, utilizzato come secondo approccio di confronto.

Per entrambi i modelli ho usato lo stesso preprocessing:

1. ho trattato come valori mancanti gli zeri non plausibili in alcune misure cliniche;
2. ho sostituito i valori mancanti con la mediana calcolata sul training set;
3. ho standardizzato le variabili numeriche;
4. ho valutato i modelli con accuracy, precision, recall, F1-score e ROC-AUC.

La Logistic Regression è il modello salvato e utilizzato dall'applicazione. KNN rimane nel codice perché il confronto tra modelli è parte dell'esperimento documentato nel notebook.

## Struttura del repository

```text
app/
├── app.py                  # applicazione Flask, pagine web e API JSON
├── templates/index.html    # interfaccia HTML del modulo di predizione
└── static/style.css        # stile dell'interfaccia

src/
├── config.py               # percorsi, colonne e configurazione condivisa
├── data.py                 # lettura, download e validazione del dataset
├── eda.py                  # generazione opzionale dei grafici EDA
└── train.py                # training, confronto e salvataggio del modello

notebooks/
├── 01_Analisi_Dati.ipynb   # analisi esplorativa e visualizzazioni
└── 02_Machine_Learning.ipynb # esperimenti, metriche e scelta del modello

tests/test_app.py            # test automatici dell'applicazione e dell'API
data/raw/                    # dataset locale, escluso da Git
models/                      # modello addestrato, escluso da Git
reports/                     # metriche e grafici generati, esclusi da Git
Dockerfile                   # immagine Docker dell'applicazione
docker-compose.yml           # avvio del servizio e volumi condivisi
requirements.txt             # dipendenze Python
```

## Flusso del progetto

I notebook documentano l'analisi e gli esperimenti, mentre lo script `src/train.py` esegue il training operativo usato per creare il modello dell'applicazione.

```text
dataset → preprocessing → Logistic Regression/KNN
       → valutazione → modello Logistic Regression salvato
       → Flask/API → predizione sui dati inseriti
```

Il modello viene salvato in `models/diabetes_model.joblib` insieme a tutto il preprocessing della pipeline. In questo modo l'applicazione applica agli input dell'utente le stesse trasformazioni utilizzate durante l'addestramento.

## Installazione e avvio locale

Sono richiesti Python 3.11 o superiore e `pip`.

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

### Addestramento

Per addestrare il modello ed elaborare le metriche:

```bash
python -m src.train
```

Se `data/raw/diabetes.csv` non è presente, `src.data` prova a scaricarlo automaticamente dalla fonte pubblica. È anche possibile inserire il CSV manualmente nella stessa cartella. Il comando aggiorna:

- `models/diabetes_model.joblib`, il modello usato da Flask;
- `reports/metrics.json`, il confronto tra i modelli e le metriche ottenute.

Per rigenerare i grafici dell'analisi esplorativa:

```bash
python -m src.eda
```

### Avvio dell'applicazione

```bash
flask --app app.app run --debug
```

L'interfaccia è disponibile all'indirizzo <http://127.0.0.1:5000>.

L'applicazione non esegue un nuovo training a ogni richiesta: carica il modello già salvato. Se il dataset viene modificato, è necessario eseguire nuovamente `python -m src.train` per creare un nuovo modello.

### Test

```bash
pytest -q
```

## API JSON

L'endpoint `POST /api/predict` riceve gli otto predittori numerici e restituisce la classe stimata, la probabilità e un messaggio esplicativo.

```bash
curl -X POST http://127.0.0.1:5000/api/predict \\
  -H 'Content-Type: application/json' \\
  -d '{"pregnancies":2,"glucose":120,"blood_pressure":70,"skin_thickness":20,"insulin":79,"bmi":28.5,"diabetes_pedigree":0.3,"age":35}'
```

L'endpoint `GET /api/health` indica se l'applicazione è attiva e se il modello è disponibile.

## Docker

Docker crea un ambiente riproducibile installando le dipendenze e avviando Flask tramite Gunicorn.

Prima di avviare il container è consigliabile preparare localmente dataset e modello:

```bash
source .venv/bin/activate
python -m src.train
```

Poi si può costruire l'immagine e avviare il servizio:

```bash
docker compose up --build
```

Il servizio è disponibile su <http://localhost:5000>.

Il file `docker-compose.yml` collega le cartelle locali `data/`, `models/` e `reports/` alle corrispondenti cartelle del container. In questo modo il container riutilizza il dataset e il modello già presenti. Se il modello non esiste, il comando di avvio tenta di eseguire automaticamente `src.train`; se il modello esiste già, non viene sovrascritto automaticamente.

Per aggiornare il modello dopo una modifica del dataset:

```bash
docker compose run --rm web python -m src.train
```

Per fermare il servizio:

```bash
docker compose down
```

## Dataset e limiti

Il dataset PIMA Indians Diabetes contiene 768 osservazioni e descrive una popolazione specifica di donne adulte di origine Pima. Di conseguenza, i risultati non possono essere considerati automaticamente validi per popolazioni diverse.

Il recall ottenuto mostra che una parte dei casi positivi non viene riconosciuta. Inoltre, il dataset è relativamente piccolo e contiene valori mancanti rappresentati da zeri. Le prestazioni misurate sono quindi utili per valutare l'esperimento, ma non costituiscono una validazione clinica.

Possibili sviluppi futuri sono l'ottimizzazione controllata dei parametri `C` della Logistic Regression e `k` di KNN, la scelta di una soglia diversa da 0,5 per privilegiare il recall, la validazione su dati indipendenti e l'analisi dell'interpretabilità e dei possibili bias.

## Git e riproducibilità

Il repository contiene codice, notebook, test e documentazione. Dataset, modello addestrato, grafici generati, ambienti virtuali e file temporanei sono esclusi tramite `.gitignore`.

Per le modifiche si usano commit piccoli e descrittivi, con verbo all'imperativo e una motivazione quando utile:

```bash
git add <file>
git commit -m "docs: clarify Docker workflow"
git push origin main
```

Prefissi consigliati: `feat`, `fix`, `docs`, `test`, `refactor` e `chore`.
