# Diabetes Risk Prediction

Progetto finale per il corso **Laboratorio di Intelligenza Artificiale Applicata**.
L'applicazione confronta modelli di machine learning per stimare la probabilita` di diabete usando il dataset PIMA Indians Diabetes e fornisce sia una web interface Flask sia un'API JSON.

> **Avvertenza:** il progetto ha esclusivamente finalita` didattiche. Non e` un dispositivo medico e non deve essere usato per diagnosi, decisioni cliniche o sostituire un professionista sanitario.

## Funzionalita`

- Analisi esplorativa riproducibile e grafici salvati in `reports/figures/`.
- Preprocessing dei valori clinicamente implausibili, imputazione mediana e standardizzazione.
- Confronto tra Logistic Regression, Random Forest e SVM, con ottimizzazione tramite cross-validation.
- Valutazione su test set separato: accuracy, precision, recall, F1 e ROC-AUC.
- Interfaccia web per le predizioni e endpoint REST `POST /api/predict`.
- Docker e Docker Compose per un avvio riproducibile.

## Struttura

```text
app/                  applicazione Flask, template e stile
src/                  download dati, EDA, preprocessing e training
data/raw/             dataset scaricato localmente (ignorato da Git)
models/               modello addestrato (ignorato da Git)
reports/              metriche e grafici generati (ignorati da Git)
tests/                test automatici
```

## Avvio locale

Sono richiesti Python 3.11+ e pip.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python -m src.train              # scarica i dati e addestra il modello
python -m src.eda                # genera i grafici dell'analisi
flask --app app.app run --debug
```

Apri `http://127.0.0.1:5000`. Per eseguire i test: `pytest`.

## API

```bash
curl -X POST http://127.0.0.1:5000/api/predict \\
  -H 'Content-Type: application/json' \\
  -d '{"pregnancies":2,"glucose":120,"blood_pressure":70,"skin_thickness":20,"insulin":79,"bmi":28.5,"diabetes_pedigree":0.3,"age":35}'
```

La risposta contiene la classe stimata (`prediction`), la probabilita` stimata e un messaggio. I campi devono essere numerici; eta`, BMI e numero di gravidanze non possono essere negativi.

## Docker

```bash
docker compose up --build
```

Al primo avvio il container scarica i dati e addestra il modello; l'app sara` poi disponibile su `http://localhost:5000`.

## Dataset e limiti

Il dataset PIMA Indians Diabetes contiene record clinici di donne adulte di origine Pima. Questo limita la generalizzabilita` del modello: la popolazione non rappresenta tutte le persone e i dati riflettono il contesto storico della raccolta. Le variabili con valore zero non plausibile (glucosio, pressione, spessore cutaneo, insulina e BMI) sono trattate come mancanti.

La scelta del modello finale e` guidata dalla ROC-AUC sul validation set in cross-validation, mentre le metriche finali vengono calcolate una sola volta su un test set separato. Le prestazioni, quindi, non devono essere interpretate come validazione clinica.

## Git workflow

Ogni modifica logica va in un commit piccolo e leggibile, ad esempio:

```bash
git add src/data.py src/train.py requirements.txt
git commit -m "feat: add reproducible training pipeline"
git push origin main
```

Prefissi consigliati: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`. Non committare dataset, modelli, ambienti virtuali, chiavi o file `.env`.
