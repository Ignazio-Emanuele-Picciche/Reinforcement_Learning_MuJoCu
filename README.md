# Reinforcement_Learning_MuJoCu

<p align="center">
  <img src="https://github.com/Ignazio-Emanuele-Picciche/Reinforcement_Learning_Ant_MuJoCu/blob/main/Test%20Cheetah/videos/halfcheetah.gif" alt="HalfCheetah GIF">
</p>

To read the README.md in English, click [here](README_english.md).

## Indice 
1. [Descrizione del progetto](#descrizione-del-progetto)
2. [Setup del Enviroment](#setup-del-enviroment)
3. [Struttura del repository](#struttura-del-repository)
4. [Organizzazione Progetto](#organizzazione-progetto)
4. [Grafici e Risultati](#grafici-e-risultati)
5. [Problemi affrontati e soluzioni adottate](#problemi-affrontati-e-soluzioni-adottate)

## Descrizione del progetto
Questo progetto esplora l'applicazione di algoritmi avanzati di Reinforcement Learning (RL) per l'addestramento e la valutazione di agenti autonomi nei classici ambienti HalfCheetah e Ant della libreria Gymnasium. L'obiettivo principale è ottimizzare le prestazioni degli agenti utilizzando due approcci principali per HalfCheetah: Proximal Policy Optimization (PPO) e Soft Actor-Critic (SAC). Per l'ambiente Ant, viene impiegato anche Twin Delayed Deep Deterministic Policy Gradient (TD3).

L'implementazione sfrutta Optuna per la ricerca automatizzata degli iperparametri e include ambienti personalizzati per adattare le dinamiche di apprendimento alle specifiche esigenze degli agenti. Il progetto prevede inoltre tecniche di normalizzazione per stabilizzare il training e strumenti di valutazione per analizzare in modo dettagliato le prestazioni degli agenti nei vari scenari simulati.

## Setup del Enviroment
Prima di eseguire il codice, assicurati di utilizzare la versione di Python 3.10.*. È importante prendere alcune precauzioni e configurare correttamente l'ambiente. Segui questi passaggi:

1. Creare un Ambiente Virtuale:

    -Apri il terminale o il prompt dei comandi.

    -Esegui il seguente comando per creare un ambiente virtuale chiamato "venv": python -m venv venv

2. Attivare l'Ambiente Virtuale:

    -Se stai usando Windows: `.\venv\Scripts\activate`

    -Se stai usando Unix o macOS: `source venv/bin/activate`

3. OPZIONALE - Disattivare l'Ambiente Virtuale (Quando hai finito):

    -Usa il seguente comando per disattivare l'ambiente virtuale: `deactivate`

4. Installare le Dipendenze:

    -Dopo aver clonato il progetto e attivato l'ambiente virtuale, installa le dipendenze richieste utilizzando: `pip install -r requirements.txt`
    
    Questo comando scaricherà tutti i moduli non standard richiesti dall'applicazione.

5. Se la versione di Python utilizzata per creare l'ambiente virtuale non contiene una versione aggiornata di pip, aggiorna pip utilizzando:`pip install --upgrade pip`

Una volta configurato l'ambiente virtuale e installate le dipendenze, sei pronto per eseguire l'applicazione. Semplicemente, naviga fino al file .ipynb desiderato ed eseguilo.

## Struttura del repository
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
├── prova.ipynb                        # Notebook per il tuning dei parametri  
├── README.md                          # Questo file  
├── requirements.txt                   # File con le dipendenze del progetto  
```

## Organizzazione Progetto 

Il progetto è strutturato in cartelle dedicate ai test su due ambienti di simulazione: Ant e HalfCheetah. All'interno di ciascuna di queste cartelle, i test sono ulteriormente suddivisi in base all'algoritmo di reinforcement learning utilizzato per l'addestramento del modello, tra cui PPO (Proximal Policy Optimization), SAC (Soft Actor-Critic) e TD3 (Twin Delayed Deep Deterministic Policy Gradient).

Ogni cartella dedicata a un algoritmo contiene i seguenti notebook Jupyter:

1. Hyperparameter Tuning: un notebook specifico per la regolazione degli iperparametri, utile per ottimizzare le prestazioni del modello prima della fase di addestramento effettivo.

2. Training: il notebook principale per l'addestramento del modello nell'ambiente corrispondente, che nel caso di HalfCheetah include anche la visualizzazione di grafici riassuntivi delle metriche di addestramento.

3. Rendering: un notebook dedicato alla visualizzazione e alla valutazione del comportamento del modello dopo l'addestramento.

Nel caso specifico del notebook di rendering del modello PPO per l'ambiente HalfCheetah, sono disponibili due modalità di rendering:

1. Human Mode: questa modalità consente di visualizzare in tempo reale l'interazione dell'agente con l'ambiente, rendendolo adatto per l'osservazione diretta da parte dell'utente.
2. RGB Array Mode: questa modalità restituisce i frame del rendering sotto forma di array di valori RGB, permettendo di elaborare e salvare il video per analisi successive o per la creazione di animazioni.
Questa organizzazione consente una gestione chiara e modulare degli esperimenti, facilitando l'analisi delle performance dei diversi algoritmi applicati agli ambienti di simulazione.

## Grafici e Risultati
Per capire quanto bene ha imparato il modello, il progetto include un confronto tra la policy addestrata e una policy casuale, oltre a un grafico che mostra l’andamento della reward media durante l’addestramento.

1. Confronto tra policy addestrata e casuale
Dopo l’addestramento, il modello viene testato e confrontato con un agente che si muove in modo casuale (random policy). Questo serve a vedere quanto l’algoritmo ha migliorato il comportamento rispetto a un’azione priva di apprendimento. Idealmente, la policy addestrata dovrebbe mostrare movimenti più fluidi ed efficienti rispetto alla policy casuale.

2. Grafico della reward media nel tempo
Durante l’addestramento, viene registrata la reward media ottenuta dal modello a intervalli di tempo. Questo grafico aiuta a capire come sta migliorando il modello nel tempo:

    -Se la reward sale, significa che il modello sta imparando a muoversi meglio.
    
    -Se la reward si stabilizza o scende, potrebbe indicare un problema o che il modello ha raggiunto il suo massimo livello di apprendimento.

Questi strumenti aiutano a valutare la qualità dell’addestramento e a confrontare i diversi algoritmi usati.

Esempio grafici presi dal PPO Half_Cheetah:

<p align="center">
    <img src="https://github.com/Ignazio-Emanuele-Picciche/Reinforcement_Learning_Ant_MuJoCu/blob/main/Test%20Cheetah/videos/Reward.png" width="45%">
    <img src="https://github.com/Ignazio-Emanuele-Picciche/Reinforcement_Learning_Ant_MuJoCu/blob/main/Test%20Cheetah/videos/Valutazione_policy.png" width="45%">
</p>

## Problemi affrontati e soluzioni adottate
All'inizio dello sviluppo, abbiamo utilizzato un ambiente di base predefinito per le librerie, ma abbiamo riscontrato che non era adeguato né per il modello Cheetah né per Ant. In particolare, per Cheetah, l'algoritmo tendeva spesso a convergere verso soluzioni subottimali anziché trovare l'ottimo globale. Per migliorare le prestazioni, abbiamo introdotto una penalizzazione aggiuntiva qualora il dorso del Cheetah superasse un determinato angolo, incrementando progressivamente il valore della penalità per scoraggiare tali comportamenti indesiderati.

Un ulteriore miglioramento è stato ottenuto modificando l'architettura della rete neurale attraverso il parametro "policy_kwargs": dict(net_arch=[256, 256, 128]), che ha portato a risultati più soddisfacenti nel caso di Cheetah.

Per quanto riguarda Ant, abbiamo esplorato diverse strategie per migliorare le prestazioni, adattando e testando vari algoritmi con l'obiettivo di ottimizzare il controllo del modello. In particolare, abbiamo affinato l'architettura della rete neurale e regolato i parametri chiave per gestire la maggiore complessità computazionale dell'ambiente. Nonostante le difficoltà intrinseche del task, siamo riusciti a individuare e implementare le soluzioni migliori tra quelle testate, consentendo così di avanzare significativamente nell'addestramento e di ottenere risultati concreti, sebbene rimanga margine di miglioramento per arrivare a una soluzione ottimale definitiva.