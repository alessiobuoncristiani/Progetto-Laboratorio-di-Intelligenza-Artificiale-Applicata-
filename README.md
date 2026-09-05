# Sistema didattico per la stima del rischio di diabete

Progetto finale per il corso **Laboratorio di Intelligenza Artificiale Applicata**.

Il progetto analizza il dataset **PIMA Indians Diabetes** e sviluppa un sistema di classificazione binaria capace di stimare la probabilità che un'osservazione appartenga alla classe associata al diabete (`0` oppure `1`). Il sistema comprende analisi esplorativa, confronto e ottimizzazione di modelli, interpretabilità SHAP, analisi esplorativa dei possibili bias, confronto sperimentale con un LLM, web application Flask, API JSON, test automatici e configurazione Docker.

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
├── preprocessing.py        # trattamento riutilizzabile degli zeri non plausibili
└── train.py                # training e salvataggio del modello finale

notebooks/
├── 01_Analisi_Dati.ipynb             # analisi esplorativa e visualizzazioni
├── 02_Machine_Learning.ipynb         # modelli base e confronto iniziale
├── 03_Ottimizzazione_Modelli.ipynb   # tuning, soglie e scelta finale
├── 04_Interpretabilita_e_Bias.ipynb  # SHAP e controllo esplorativo per età
└── 05_Confronto_LLM.ipynb            # confronto sperimentale ML-Gemini

tests/
├── test_app.py              # interfaccia, validazione e API
├── test_data.py             # caricamento e schema del dataset
├── test_preprocessing.py    # trattamento dei valori non plausibili
└── test_train.py            # pipeline finale e metriche

data/raw/                    # dataset locale, escluso da Git
models/                      # modello addestrato, escluso da Git
reports/                     # metriche e grafici generati, esclusi da Git
Dockerfile                   # immagine Docker dell'applicazione
docker-compose.yml           # avvio del servizio e volumi condivisi
requirements.txt             # dipendenze del progetto e dei test
requirements-app.txt         # dipendenze minime dell'app Docker
requirements-notebook.txt    # dipendenze aggiuntive per Jupyter e Gemini
.env.example                 # esempio di configurazione senza chiavi reali
```

## Flusso del progetto

I cinque notebook hanno ruoli distinti. Il primo descrive i dati, il secondo documenta i modelli base, il terzo ottimizza i parametri e motiva la configurazione finale, il quarto usa SHAP e confronta le metriche per fasce d'età, mentre il quinto confronta in modo sperimentale il modello ML con Gemini. Lo script `src/train.py` esegue soltanto il training operativo del modello scelto, senza ripetere tuning e confronti a ogni avvio.

```text
dataset → analisi esplorativa
       → preprocessing → confronto e tuning nei notebook
       → Logistic Regression polinomiale scelta → spiegazioni SHAP
       → modello finale salvato
       → Flask/API → predizione sui dati inseriti
       → confronto sperimentale separato con LLM
```

Il modello viene salvato in `models/diabetes_model.joblib` insieme a tutto il preprocessing della pipeline. In questo modo l'applicazione applica agli input dell'utente le stesse trasformazioni utilizzate durante l'addestramento.

### Configurazione finale

La scelta è stata effettuata usando il training set e una cross-validation stratificata a 5 fold. Il test set è stato mantenuto separato fino alla valutazione finale. La configurazione selezionata ha ottenuto ROC-AUC media `0,846` nella cross-validation e, sul test set, accuracy `0,734`, precision `0,607`, recall `0,685`, F1-score `0,644` e ROC-AUC `0,825`.

Nel notebook 3 sono presenti anche il grafico dell'effetto di `C`, il confronto tra gradi polinomiali, il grafico dei diversi valori di `k`, l'analisi delle soglie e una proiezione illustrativa delle frontiere decisionali su glucosio e BMI. Nel notebook 4, SHAP mostra l'importanza globale delle variabili e spiega una singola predizione; il controllo per fasce d'età è descrittivo e non costituisce una dimostrazione di fairness o assenza di bias.

Nel notebook 5 ho confrontato il modello finale con Gemini 3.5 Flash-Lite sulle stesse 154 osservazioni di test, senza comunicare all'LLM il nome del dataset o le etichette reali. La Logistic Regression ha ottenuto accuracy `0,734`, recall `0,685` e F1-score `0,644`; Gemini ha ottenuto rispettivamente `0,714`, `0,648` e `0,614`. Il confronto è illustrativo e non costituisce una validazione diagnostica dell'LLM.

## Installazione e avvio locale

Sono richiesti Python 3.11 o superiore e `pip`.

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

Per eseguire anche i notebook in VS Code/Jupyter, incluso il confronto Gemini:

```bash
pip install -r requirements-notebook.txt
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

L'applicazione non esegue un nuovo training a ogni richiesta: carica il modello già salvato e lo mantiene in memoria per le predizioni successive. Se il dataset viene modificato, è necessario eseguire nuovamente `python -m src.train` e riavviare l'applicazione.

### Test

```bash
pytest -q
```

La suite controlla applicazione, API, gestione degli input errati, dataset, preprocessing, configurazione del modello e metriche.

## API JSON

L'endpoint `POST /api/predict` riceve gli otto predittori numerici e restituisce la classe stimata, la probabilità e un messaggio esplicativo.

```bash
curl -X POST http://127.0.0.1:5000/api/predict \\
  -H 'Content-Type: application/json' \\
  -d '{"pregnancies":2,"glucose":120,"blood_pressure":70,"skin_thickness":20,"insulin":79,"bmi":28.5,"diabetes_pedigree":0.3,"age":35}'
```

L'endpoint `GET /api/health` indica se l'applicazione è attiva e se il modello è disponibile.

L'interfaccia include anche un pulsante per compilare un esempio e avvisi non bloccanti per valori fuori dagli intervalli tipici osservati nel dataset. Se glucosio, pressione, spessore cutaneo, insulina o BMI valgono zero, l'applicazione chiarisce che il modello li considera dati mancanti e li sostituisce con la mediana del training set. Questi controlli aiutano a individuare errori di digitazione, ma non sono una validazione clinica.

Dopo una predizione è disponibile una breve simulazione *what-if*: variando glucosio e BMI con due slider, l'app invia una nuova richiesta all'API e mostra la relativa stima, una barra di probabilità e la differenza in punti percentuali rispetto al risultato iniziale. Gli altri dati rimangono invariati, il modello non viene riaddestrato e il risultato originario non viene modificato. La funzione serve soltanto a esplorare il comportamento del modello e non a suggerire cambiamenti terapeutici.

## Confronto opzionale con Gemini

Il notebook 5 usa una chiave Gemini soltanto per ripetere l'esperimento. Copio il file di esempio e inserisco la chiave nel nuovo `.env`, che rimane locale ed è escluso sia da Git sia dall'immagine Docker:

```bash
cp .env.example .env
```

Nel notebook le chiamate sono disattivate per impostazione predefinita con `RUN_API_CALLS = False`. Per ripetere l'esperimento imposto temporaneamente il valore a `True`, eseguo le chiamate, poi lo riporto a `False` prima di salvare. I risultati completi restano visibili negli output del notebook; i CSV locali sono ignorati da Git.

## Docker

Docker crea un ambiente riproducibile installando soltanto le dipendenze operative elencate in `requirements-app.txt` e avviando Flask tramite Gunicorn. Il server usa un worker a thread (`gthread`, quattro thread) e un keep-alive breve: in questo modo una connessione browser rimasta aperta non blocca le richieste successive. Un healthcheck interroga periodicamente `/api/health` per verificare che il servizio risponda.

### Avvio rapido con Docker

È sufficiente avere Docker Desktop installato e avviato. Non è necessario installare Python, Jupyter o le librerie del progetto sul computer. Dopo avere scaricato il repository, entro nella cartella e avvio il servizio:

```bash
git clone https://github.com/alessiobuoncristiani/Progetto-Laboratorio-di-Intelligenza-Artificiale-Applicata-
cd Progetto-Laboratorio-di-Intelligenza-Artificiale-Applicata-
docker compose up --build
```

Quando nel terminale compare l'indirizzo di ascolto di Gunicorn, apro nel browser:

<http://localhost:5000>

Al primo avvio, se `models/diabetes_model.joblib` non esiste, il container esegue automaticamente `python -m src.train`. Se manca anche il CSV locale, lo scarica dalla fonte pubblica prima di addestrare il modello. La prima build richiede normalmente una connessione Internet per scaricare immagine di base e dipendenze; anche il download automatico del dataset richiede la rete. Gli avvii successivi possono riutilizzare immagine, dataset e modello già salvati localmente.

Il file `docker-compose.yml` collega le cartelle locali `data/`, `models/` e `reports/` alle corrispondenti cartelle del container. In questo modo il container riutilizza il dataset e il modello già presenti. Se il modello non esiste, il comando di avvio tenta di eseguire automaticamente `src.train`; se il modello esiste già, non viene sovrascritto automaticamente.

Per aggiornare il modello dopo una modifica del dataset:

```bash
docker compose run --rm web python -m src.train
```

Per fermare il servizio:

```bash
docker compose down
```

Se `docker compose up --build` è in esecuzione in primo piano, posso prima interrompere la visualizzazione dei log con `Ctrl+C` e poi usare `docker compose down`. I dati scaricati e il modello restano nelle cartelle locali e non vengono cancellati.

## Dataset e limiti

Il dataset PIMA Indians Diabetes contiene 768 osservazioni e descrive una popolazione specifica di donne adulte di origine Pima. Di conseguenza, i risultati non possono essere considerati automaticamente validi per popolazioni diverse.

Il recall ottenuto mostra che una parte dei casi positivi non viene riconosciuta. Inoltre, il dataset è relativamente piccolo e contiene valori mancanti rappresentati da zeri. Le prestazioni misurate sono quindi utili per valutare l'esperimento, ma non costituiscono una validazione clinica.

Nel notebook 3 ho ottimizzato `C`, grado polinomiale, bilanciamento delle classi e `k` di KNN, oltre ad aver analizzato soglie diverse. Nel notebook 4 ho aggiunto spiegazioni SHAP e un primo confronto descrittivo per fasce d'età; nel notebook 5 ho svolto il confronto opzionale con un LLM. Restano possibili sviluppi la raccolta di più dati, la validazione su una popolazione indipendente e una valutazione della fairness con gruppi e numerosità adeguati.

### Tentativo di integrazione SHAP nell'applicazione

Ho provato a mostrare una spiegazione SHAP anche dopo ogni predizione dell'interfaccia web. L'approccio model-agnostic usato nel notebook, però, richiede l'inizializzazione di Numba e LLVM e in Docker poteva bloccare il worker Gunicorn fino al timeout, causando risposte lente o il mancato caricamento del CSS. Ho quindi scelto di non includere questa funzionalità nell'applicazione operativa: l'interpretabilità SHAP rimane documentata nel notebook 4, dove può essere analizzata senza compromettere l'affidabilità del servizio web.

## Git e riproducibilità

Il repository contiene codice, notebook, test e documentazione. Dataset, modello addestrato, grafici generati, risultati CSV dell'LLM, ambienti virtuali, chiavi API e file temporanei sono esclusi tramite `.gitignore` e `.dockerignore`; gli artefatti possono essere rigenerati seguendo le istruzioni precedenti.

Per le modifiche si usano commit piccoli e descrittivi, con verbo all'imperativo e una motivazione quando utile:

```bash
git add <file>
git commit -m "docs: clarify Docker workflow"
git push origin main
```

Prefissi consigliati: `feat`, `fix`, `docs`, `test`, `refactor` e `chore`.
