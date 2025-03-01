# Reinforcement_Learning_MuJoCu

<p align="center">
    <img src="Test Cheetah/media/halfcheetah.gif" alt="HalfCheetah GIF">
</p>

<!-- To read the README.md in English, click [here](myLib/README_english.md). -->

## Indice 
1. [Descrizione del progetto](#1-descrizione-del-progetto)
2. [Setup del Enviroment](#2-setup-del-enviroment)
3. [Struttura del repository](#3-struttura-del-repository)
4. [Organizzazione Progetto](#4-organizzazione-progetto)
4. [Grafici e Risultati](#5-grafici-e-risultati)
5. [Problemi affrontati e soluzioni adottate](#6-problemi-affrontati-e-soluzioni-adottate)

## 1. Descrizione del progetto
Questo progetto esplora l'applicazione di algoritmi avanzati di Reinforcement Learning (RL) per l'addestramento e la valutazione di agenti autonomi nei classici ambienti HalfCheetah e Ant della libreria Gymnasium. L'obiettivo principale è ottimizzare le prestazioni degli agenti utilizzando due approcci principali per HalfCheetah: Proximal Policy Optimization (PPO) e Soft Actor-Critic (SAC). Per l'ambiente Ant, viene impiegato anche Twin Delayed Deep Deterministic Policy Gradient (TD3).

L'implementazione sfrutta Optuna per la ricerca automatizzata degli iperparametri e include ambienti personalizzati per adattare le dinamiche di apprendimento alle specifiche esigenze degli agenti. Il progetto prevede inoltre tecniche di normalizzazione per stabilizzare il training e strumenti di valutazione per analizzare in modo dettagliato le prestazioni degli agenti nei vari scenari simulati.

## 2. Setup del Enviroment
Prima di eseguire il codice, assicurati di utilizzare la versione di Python 3.10.*. È importante prendere alcune precauzioni e configurare correttamente l'ambiente. Segui questi passaggi:

1. Creare un Ambiente Virtuale:

    - Apri il terminale o il prompt dei comandi.

    - Esegui il seguente comando per creare un ambiente virtuale chiamato "venv": python -m venv venv

2. Attivare l'Ambiente Virtuale:

    - Se stai usando Windows: `.\venv\Scripts\activate`

    - Se stai usando Unix o macOS: `source venv/bin/activate`

3. OPZIONALE - Disattivare l'Ambiente Virtuale (Quando hai finito):

    - Usa il seguente comando per disattivare l'ambiente virtuale: `deactivate`

4. Installare le Dipendenze:

    - Dopo aver clonato il progetto e attivato l'ambiente virtuale, installa le dipendenze richieste utilizzando: `pip install -r requirements.txt`
    
    - Questo comando scaricherà tutti i moduli non standard richiesti dall'applicazione.

5. Se la versione di Python utilizzata per creare l'ambiente virtuale non contiene una versione aggiornata di pip, aggiorna pip utilizzando:`pip install --upgrade pip`

Una volta configurato l'ambiente virtuale e installate le dipendenze, sei pronto per eseguire l'applicazione. Semplicemente, naviga fino al file .ipynb desiderato ed eseguilo.

## 3. Struttura del repository
Il progetto è organizzato nelle seguenti directory e file:
```plaintext
REINFORCEMENT_LEARNING_MUJOCU/
├── Docs/                           # Documentazione e risultati  
│   ├── Best Rewards.xlsx           # Foglio Excel con i miglior risultati ottenuti  
│   ├── Spiegazione metriche RL.docx # Documento con spiegazione delle metriche usate  
│
├── Test Ant/                        # Esperimenti con l'ambiente Ant  
│   ├── PPO/                          # Esperimenti con PPO  
│   │   ├── logs/                     # Log di addestramento  
│   │   ├── ppo_Ant_tensorboard/      # File di TensorBoard  
│   │   ├── ANT_train_PPO.ipynb       # Notebook per l'addestramento  
│   │   ├── HT_Ant_test_PPO.ipynb     # Notebook per il tuning dei parametri  
│   │   ├── Render_ANT_PPO.ipynb      # Notebook per il rendering del modello addestrato  
│   │   ├── ppo_Ant_model_PPO16.zip   # Modello salvato  
│   │   ├── ppo_Ant_model_PPO17_senza_hp.zip  # Modello senza hyperparameter tuning  
│   │   ├── ppo_Ant_model_PPO18.zip   # Ultima versione del modello  
│   │   ├── vecnormalize_Ant.pkl      # File per la normalizzazione degli input  
│   │  
│   ├── SAC/                          # Esperimenti con SAC  
│   │   ├── logs/                     # Log di addestramento  
│   │   ├── sac_Ant_tensorboard/      # File di TensorBoard  
│   │   ├── ANT_train_SAC.ipynb       # Notebook per l'addestramento  
│   │   ├── HT_Ant_test_SAC.ipynb     # Notebook per il tuning dei parametri    
│   │   ├── Render_ANT_SAC.ipynb      # Notebook per il rendering  
│   │   ├── sac_Ant_model.zip         # Modello salvato  
│   │   ├── vecnormalize_Ant.pkl      # File per la normalizzazione degli input  
│   │  
│   ├── TD3/                          # Esperimenti con TD3  
│   │   ├── td3_Ant_tensorboard/      # File di TensorBoard  
│   │   ├── ANT_train_TD3.ipynb       # Notebook per l'addestramento  
│   │   ├── HT_Ant_test_TD3.ipynb     # Notebook per il tuning dei parametri   
│   │   ├── Render_ANT_TD3.ipynb      # Notebook per il rendering  
│
├── Test Cheetah/                     # Esperimenti con l'ambiente HalfCheetah  
│   ├── PPO_CustomENV/                # Esperimenti con PPO  
│   │   ├── logs/                     # Log di addestramento  
│   │   ├── ppo_HalfCheetah_tensorboard/ # File di TensorBoard  
│   │   ├── HT_HalfCheetah_ppo.ipynb  # Notebook per il tuning dei parametri   
│   │   ├── ppo_cheetah.ipynb         # Notebook di addestramento  
│   │   ├── Render_HalfCheetah_ppo.ipynb # Notebook per il rendering  
│   │   ├── ppo_HalfCheetah_model.zip # Modello salvato  
│   │   ├── vecnormalize_HalfCheetah.pkl # File per la normalizzazione  
│   │  
│   ├── SAC_CustomENV/                # Esperimenti con SAC  
│   │   ├── logs/                     # Log di addestramento  
│   │   ├── sac_HalfCheetah_tensorboard/ # File di TensorBoard  
│   │   ├── HT_HalfCheetah_sac.ipynb  # Notebook per il tuning dei parametri   
│   │   ├── sac_cheetah.ipynb         # Notebook di addestramento  
│   │   ├── Render_HalfCheetah_sac.ipynb # Notebook per il rendering  
│   │   ├── sac_HalfCheetah_model.zip # Modello salvato  
│   │   ├── vecnormalize_HalfCheetah.pkl # File per la normalizzazione  
│
├── videos/                           # Cartella con video delle migliori policy  
│   ├── halfcheetah_best_policy.mp4   # Video della migliore policy  
│   ├── halfcheetah.gif               # GIF della simulazione  
│
├── venv/                             # Ambiente virtuale Python  
│
├── .gitignore                        # File per ignorare file non necessari su Git  
├── README.md                          # Questo file  
├── requirements.txt                   # File con le dipendenze del progetto  
```

## 4. Organizzazione Progetto 

Il progetto è strutturato in modo da facilitare la gestione e l'analisi degli esperimenti di reinforcement learning su due ambienti di simulazione: Ant e HalfCheetah. Ogni ambiente ha una propria cartella dedicata, all'interno della quale i test sono ulteriormente suddivisi in base all'algoritmo di reinforcement learning utilizzato per l'addestramento del modello. Gli algoritmi principali utilizzati sono:

- PPO (Proximal Policy Optimization)
- SAC (Soft Actor-Critic)
- TD3 (Twin Delayed Deep Deterministic Policy Gradient)

### Struttura delle Cartelle

Ogni cartella dedicata a un algoritmo contiene i seguenti notebook Jupyter, ciascuno con uno scopo specifico:

1. **Hyperparameter Tuning**: Questo notebook è dedicato alla regolazione degli iperparametri. Ottimizzare gli iperparametri è cruciale per migliorare le prestazioni del modello prima della fase di addestramento effettivo. Utilizziamo strumenti come Optuna per automatizzare questo processo e trovare le migliori configurazioni possibili.

2. **Training**: Il notebook principale per l'addestramento del modello nell'ambiente corrispondente. In questo notebook, il modello viene addestrato utilizzando l'algoritmo specificato. Per l'ambiente HalfCheetah, il notebook include anche la visualizzazione di grafici riassuntivi delle metriche di addestramento, come la reward media nel tempo, per monitorare i progressi del modello.

3. **Rendering**: Un notebook dedicato alla visualizzazione e alla valutazione del comportamento del modello dopo l'addestramento. Questo notebook permette di osservare come l'agente interagisce con l'ambiente e di valutare qualitativamente le sue prestazioni.

### Modalità di Rendering

Nel caso specifico del notebook di rendering del modello PPO per l'ambiente HalfCheetah, sono disponibili due modalità di rendering:

1. **Human Mode**: Questa modalità consente di visualizzare in tempo reale l'interazione dell'agente con l'ambiente. È particolarmente utile per l'osservazione diretta da parte dell'utente, permettendo di vedere immediatamente come l'agente si comporta.

2. **RGB Array Mode**: Questa modalità restituisce i frame del rendering sotto forma di array di valori RGB. Questo formato è utile per elaborare e salvare il video per analisi successive o per la creazione di animazioni. Può essere utilizzato per generare GIF o video che mostrano l'evoluzione del comportamento dell'agente nel tempo.

### Vantaggi dell'Organizzazione

Questa organizzazione modulare consente una gestione chiara e strutturata degli esperimenti. Ogni fase del processo di reinforcement learning, dalla regolazione degli iperparametri all'addestramento e alla valutazione, è separata in notebook distinti. Questo approccio facilita l'analisi delle performance dei diversi algoritmi applicati agli ambienti di simulazione e permette di identificare rapidamente le aree di miglioramento.

Inoltre, la suddivisione in cartelle e notebook specifici rende il progetto facilmente navigabile e comprensibile, anche per chi non ha familiarità con il codice. Ogni notebook è documentato e contiene spiegazioni dettagliate dei passaggi eseguiti, rendendo il progetto accessibile e utile per scopi educativi e di ricerca.

## 5. Grafici e Risultati
Per capire quanto bene ha imparato il modello, il progetto include un confronto tra la policy addestrata e una policy casuale, oltre a un grafico che mostra l’andamento della reward media durante l’addestramento.

1. Confronto tra policy addestrata e casuale
Dopo l’addestramento, il modello viene testato e confrontato con un agente che si muove in modo casuale (random policy). Questo serve a vedere quanto l’algoritmo ha migliorato il comportamento rispetto a un’azione priva di apprendimento. Idealmente, la policy addestrata dovrebbe mostrare movimenti più fluidi ed efficienti rispetto alla policy casuale.

2. Grafico della reward media nel tempo
Durante l’addestramento, viene registrata la reward media ottenuta dal modello a intervalli di tempo. Questo grafico aiuta a capire come sta migliorando il modello nel tempo:

    - Se la reward sale, significa che il modello sta imparando a muoversi meglio.
    
    - Se la reward si stabilizza o scende, potrebbe indicare un problema o che il modello ha raggiunto il suo massimo livello di apprendimento.

Questi strumenti aiutano a valutare la qualità dell’addestramento e a confrontare i diversi algoritmi usati.

Esempio grafici presi dal PPO Half_Cheetah:

<p align="center">
    <img src="Test Cheetah/media/Reward.png" width="45%">
    <img src="Test Cheetah/media/Valutazione_policy.png" width="45%">
</p>

## 6. Problemi affrontati e soluzioni adottate
All'inizio dello sviluppo, abbiamo utilizzato un ambiente di base predefinito per le librerie, ma abbiamo riscontrato che non era adeguato né per il modello Cheetah né per Ant. In particolare, per Cheetah, l'algoritmo tendeva spesso a convergere verso soluzioni subottimali anziché trovare l'ottimo globale. Per migliorare le prestazioni, abbiamo introdotto una penalizzazione aggiuntiva qualora il dorso del Cheetah superasse un determinato angolo, incrementando progressivamente il valore della penalità per scoraggiare tali comportamenti indesiderati.

Un ulteriore miglioramento è stato ottenuto modificando l'architettura della rete neurale attraverso il parametro "policy_kwargs": dict(net_arch=[256, 256, 128]), che ha portato a risultati più soddisfacenti nel caso di Cheetah.

Per quanto riguarda Ant, abbiamo esplorato diverse strategie per migliorare le prestazioni, adattando e testando vari algoritmi con l'obiettivo di ottimizzare il controllo del modello. In particolare, abbiamo affinato l'architettura della rete neurale e regolato i parametri chiave per gestire la maggiore complessità computazionale dell'ambiente. Nonostante le difficoltà intrinseche del task, siamo riusciti a individuare e implementare le soluzioni migliori tra quelle testate, consentendo così di avanzare significativamente nell'addestramento e di ottenere risultati concreti, sebbene rimanga margine di miglioramento per arrivare a una soluzione ottimale definitiva.