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

La configurazione finale è una Logistic Regression polinomiale di grado 2 con `C=0,1`, penalizzazione L2, `class_weight='balanced'` e soglia decisionale `0,5`. KNN è utilizzato esclusivamente negli esperimenti documentati nei notebook, dove il confronto tra modelli motiva la scelta finale.

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
└── train.py                # training e salvataggio del modello finale

notebooks/
├── 01_Analisi_Dati.ipynb   # analisi esplorativa e visualizzazioni
├── 02_Machine_Learning.ipynb # modelli base, metriche e confronto iniziale
└── 03_Ottimizzazione_Modelli.ipynb # tuning, soglie e frontiere decisionali

tests/test_app.py            # test automatici dell'applicazione e dell'API
data/raw/                    # dataset locale, escluso da Git
models/                      # modello addestrato, escluso da Git
reports/                     # metriche e grafici generati, esclusi da Git
Dockerfile                   # immagine Docker dell'applicazione
docker-compose.yml           # avvio del servizio e volumi condivisi
requirements.txt             # dipendenze Python
```

## Flusso del progetto

I tre notebook hanno ruoli distinti. Il primo descrive i dati, il secondo documenta i modelli base e il terzo ottimizza i parametri e motiva la configurazione finale. Lo script `src/train.py` esegue soltanto il training operativo del modello scelto, senza ripetere il confronto con KNN a ogni avvio.

```text
dataset → analisi esplorativa
       → preprocessing → confronto e tuning nei notebook
       → Logistic Regression polinomiale scelta
       → modello finale salvato
       → Flask/API → predizione sui dati inseriti
```

Il modello viene salvato in `models/diabetes_model.joblib` insieme a tutto il preprocessing della pipeline. In questo modo l'applicazione applica agli input dell'utente le stesse trasformazioni utilizzate durante l'addestramento.

### Configurazione finale

La scelta è stata effettuata usando il training set e una cross-validation stratificata a 5 fold. Il test set è stato mantenuto separato fino alla valutazione finale. La configurazione selezionata ha ottenuto ROC-AUC media `0,846` nella cross-validation e, sul test set, accuracy `0,734`, precision `0,607`, recall `0,685`, F1-score `0,644` e ROC-AUC `0,825`.

Nel notebook 3 sono presenti anche il grafico dell'effetto di `C`, il confronto tra gradi polinomiali, il grafico dei diversi valori di `k`, l'analisi delle soglie e una proiezione illustrativa delle frontiere decisionali su glucosio e BMI.

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
- `reports/metrics.json`, la configurazione scelta e le metriche finali sul test set.

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

Nel notebook 3 ho già ottimizzato `C`, il grado polinomiale, il bilanciamento delle classi e `k` di KNN, oltre ad aver analizzato soglie diverse. Restano possibili sviluppi la raccolta di più dati, la validazione su una popolazione indipendente, l'analisi dell'interpretabilità e lo studio dei possibili bias.

## Git e riproducibilità

Il repository contiene codice, notebook, test e documentazione. Dataset, modello addestrato, grafici generati, ambienti virtuali e file temporanei sono esclusi tramite `.gitignore`; possono essere rigenerati seguendo le istruzioni precedenti.

Per le modifiche si usano commit piccoli e descrittivi, con verbo all'imperativo e una motivazione quando utile:

```bash
git add <file>
git commit -m "docs: clarify Docker workflow"
git push origin main
```

Prefissi consigliati: `feat`, `fix`, `docs`, `test`, `refactor` e `chore`.
