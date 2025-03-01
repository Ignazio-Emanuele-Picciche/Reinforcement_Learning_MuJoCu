# Reinforcement_Learning_MuJoCu

<p align="center">
    <img src="Test Cheetah/media/halfcheetah.gif" alt="HalfCheetah GIF">
    <img src="myLib/ANT_media/rendering_ant_ppo15.gif" alt="Ant GIF">
</p>

<!-- To read the README.md in English, click [here](myLib/README_english.md). -->

## Indice 
1. [Descrizione del progetto](#1-descrizione-del-progetto)
2. [Setup del Enviroment](#2-setup-del-enviroment)
3. [Struttura del repository](#3-struttura-del-repository)
4. [Organizzazione Progetto](#4-organizzazione-progetto)
    - [Struttura delle Cartelle](#41-struttura-delle-cartelle)
    - [Modalità di Rendering](#42-modalità-di-rendering)
    - [Vantaggi dell'Organizzazione](#43-vantaggi-dellorganizzazione)

5. [Grafici e Risultati](#5-grafici-e-risultati)
    - [Confronto tra Policy Addestrata e Casuale](#51-confronto-tra-policy-addestrata-e-casuale)
    - [Grafico della Reward Media nel Tempo](#52-grafico-della-reward-media-nel-tempo)
    - [Grafici](#53-grafici)

6. [Problemi affrontati e soluzioni adottate](#6-problemi-affrontati-e-soluzioni-adottate)
    - [Problemi e soluzioni con il modello Cheetah](#61-problemi-con-il-modello-cheetah)
    - [Problemi e soluzioni con il modello Ant](#62-problemi-con-il-modello-ant)

## 1. Descrizione del progetto
Questo progetto esplora l'applicazione di algoritmi avanzati di Reinforcement Learning (RL) per l'addestramento e la valutazione di agenti autonomi nei classici ambienti HalfCheetah e Ant della libreria Gymnasium. L'obiettivo principale è ottimizzare le prestazioni degli agenti utilizzando tre approcci principali:

- **Proximal Policy Optimization (PPO)**
- **Soft Actor-Critic (SAC)**
- **Twin Delayed Deep Deterministic Policy Gradient (TD3)**

L'implementazione sfrutta Optuna per la ricerca automatizzata degli iperparametri, ottimizzando le configurazioni per massimizzare le prestazioni degli agenti. Inoltre, include ambienti personalizzati per adattare le dinamiche di apprendimento alle specifiche esigenze degli agenti, migliorando la robustezza e l'efficacia del training.

Il progetto prevede tecniche di normalizzazione per stabilizzare il training, riducendo la varianza delle osservazioni e migliorando la convergenza degli algoritmi. Strumenti di valutazione dettagliati, come grafici della reward media e confronti tra policy addestrate e casuali, sono utilizzati per analizzare le prestazioni degli agenti nei vari scenari simulati. Questi strumenti permettono di identificare rapidamente le aree di miglioramento e di valutare l'efficacia delle diverse strategie di RL implementate.

## 2. Setup del Enviroment
Prima di eseguire il codice, assicurati di utilizzare la versione di Python 3.10.*. È importante prendere alcune precauzioni e configurare correttamente l'ambiente. Segui questi passaggi:

1. Creare un Ambiente Virtuale:

    - Apri il terminale o il prompt dei comandi.
    - Esegui il seguente comando per creare un ambiente virtuale chiamato "venv": `python -m venv venv`

2. Attivare l'Ambiente Virtuale:

    - Se stai usando Windows: `.\venv\Scripts\activate`
    - Se stai usando Unix o macOS: `source venv/bin/activate`

3. OPZIONALE - Disattivare l'Ambiente Virtuale (Quando hai finito):

    - Usa il seguente comando per disattivare l'ambiente virtuale: `deactivate`

4. Installare le Dipendenze:

    - Dopo aver clonato il progetto e attivato l'ambiente virtuale, installa le dipendenze richieste utilizzando: `pip install -r requirements.txt`
    - Questo comando scaricherà tutti i moduli non standard richiesti dall'applicazione.

5. Se la versione di Python utilizzata per creare l'ambiente virtuale non contiene una versione aggiornata di pip, aggiorna pip utilizzando: `pip install --upgrade pip`

Una volta configurato l'ambiente virtuale e installate le dipendenze, sei pronto per eseguire l'applicazione. Semplicemente, naviga fino al file .ipynb desiderato ed eseguilo.

## 3. Struttura del repository
Il repository è organizzato in modo da facilitare la navigazione e la gestione dei file e delle cartelle. La struttura principale del progetto è la seguente:

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
├── media/                           # Cartella con video/foto delle migliori policy  
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

### 4.1 Struttura delle Cartelle

Ogni cartella dedicata a un algoritmo contiene i seguenti notebook Jupyter, ciascuno con uno scopo specifico:

1. **Hyperparameter Tuning**: Questo notebook è dedicato alla regolazione degli iperparametri. Ottimizzare gli iperparametri è cruciale per migliorare le prestazioni del modello prima della fase di addestramento effettivo. Utilizziamo strumenti come Optuna per automatizzare questo processo e trovare le migliori configurazioni possibili.

2. **Training**: Il notebook principale per l'addestramento del modello nell'ambiente corrispondente. In questo notebook, il modello viene addestrato utilizzando l'algoritmo specificato. Per l'ambiente HalfCheetah, il notebook include anche la visualizzazione di grafici riassuntivi delle metriche di addestramento, come la reward media nel tempo, per monitorare i progressi del modello.

3. **Rendering**: Un notebook dedicato alla visualizzazione e alla valutazione del comportamento del modello dopo l'addestramento. Questo notebook permette di osservare come l'agente interagisce con l'ambiente e di valutare qualitativamente le sue prestazioni.

### 4.2 Modalità di Rendering

Nel caso specifico del notebook di rendering del modello PPO per l'ambiente HalfCheetah, sono disponibili due modalità di rendering:

1. **Human Mode**: Questa modalità consente di visualizzare in tempo reale l'interazione dell'agente con l'ambiente. È particolarmente utile per l'osservazione diretta da parte dell'utente, permettendo di vedere immediatamente come l'agente si comporta.

2. **RGB Array Mode**: Questa modalità restituisce i frame del rendering sotto forma di array di valori RGB. Questo formato è utile per elaborare e salvare il video per analisi successive o per la creazione di animazioni. Può essere utilizzato per generare GIF o video che mostrano l'evoluzione del comportamento dell'agente nel tempo.

### 4.3 Vantaggi dell'Organizzazione

Questa organizzazione modulare consente una gestione chiara e strutturata degli esperimenti. Ogni fase del processo di reinforcement learning, dalla regolazione degli iperparametri all'addestramento e alla valutazione, è separata in notebook distinti. Questo approccio facilita l'analisi delle performance dei diversi algoritmi applicati agli ambienti di simulazione e permette di identificare rapidamente le aree di miglioramento.

Inoltre, la suddivisione in cartelle e notebook specifici rende il progetto facilmente navigabile e comprensibile, anche per chi non ha familiarità con il codice. Ogni notebook è documentato e contiene spiegazioni dettagliate dei passaggi eseguiti, rendendo il progetto accessibile e utile per scopi educativi e di ricerca.

## 5. Grafici e Risultati
Per valutare l'efficacia dell'addestramento del modello, il progetto include un confronto tra la policy addestrata e una policy casuale, oltre a grafici che mostrano l'andamento della reward media durante l'addestramento.

### 5.1 Confronto tra Policy Addestrata e Casuale
Dopo l'addestramento, il modello viene testato e confrontato con un agente che si muove in modo casuale (random policy). Questo confronto serve a evidenziare quanto l'algoritmo abbia migliorato il comportamento rispetto a un'azione priva di apprendimento. Idealmente, la policy addestrata dovrebbe mostrare movimenti più fluidi ed efficienti rispetto alla policy casuale, dimostrando l'efficacia dell'algoritmo di reinforcement learning.

### 5.2 Grafico della Reward Media nel Tempo
Durante l'addestramento, viene registrata la reward media ottenuta dal modello a intervalli di tempo. Questo grafico è fondamentale per monitorare i progressi del modello e identificare eventuali problemi:

- **Incremento della Reward**: Se la reward media aumenta nel tempo, significa che il modello sta imparando a muoversi meglio e a ottimizzare le sue azioni.
- **Stabilizzazione o Decremento della Reward**: Se la reward media si stabilizza o diminuisce, potrebbe indicare che il modello ha raggiunto il suo massimo livello di apprendimento o che ci sono problemi nell'addestramento che necessitano di essere risolti.

### 5.2 Grafici
Di seguito sono riportati uno dei migliori risultati ottenuti durante l'addestramento con l'algoritmo PPO per l'ambiente HalfCheetah:

- **Grafico della Reward Media**: Mostra l'andamento della reward media nel tempo, evidenziando i progressi del modello durante l'addestramento.
- **Valutazione della Policy**: Confronta la performance della policy addestrata con quella di una policy casuale, dimostrando l'efficacia dell'algoritmo.

<p align="center">
    <img src="Test Cheetah/media/Reward.png" width="45%">
    <img src="Test Cheetah/media/Valutazione_policy.png" width="45%">
</p>


Di seguito ri riporta uno dei migliori risultati ottenuti durante l'addestramento con l'algoritmo PPO per l'ambiente Ant:

<p align="left">
    <img src="myLib/ANT_media/best_eval_ANT_PPO.png" width="70%">
    <img src="myLib/ANT_media/best_train_ANT_PPO.png" width="70%">
</p>

Questi strumenti di valutazione sono essenziali per comprendere la qualità dell'addestramento e per confrontare le performance dei diversi algoritmi utilizzati nel progetto.


## 6. Problemi affrontati e soluzioni adottate

Durante le prime fasi dello sviluppo, abbiamo utilizzato un ambiente di base predefinito per le librerie. Tuttavia, ci siamo presto resi conto che questo ambiente non era adeguato né per il modello Cheetah né per il modello Ant. 

### 6.1 Problemi con il modello Cheetah

Per il modello Cheetah, l'algoritmo tendeva spesso a convergere verso soluzioni subottimali anziché trovare l'ottimo globale. Questo comportamento era dovuto a una mancanza di penalizzazioni adeguate per certi movimenti indesiderati. Per migliorare le prestazioni, abbiamo introdotto una penalizzazione aggiuntiva qualora il dorso del Cheetah superasse un determinato angolo. Incrementando progressivamente il valore della penalità, siamo riusciti a scoraggiare tali comportamenti indesiderati, portando l'algoritmo a convergere verso soluzioni più ottimali.

#### Penalizzazione Personalizzata

Abbiamo implementato un wrapper personalizzato per l'ambiente `HalfCheetah-v5` che modifica la funzione di ricompensa. Questo wrapper penalizza il robot se il torso si inclina troppo all'indietro (indicando una caduta), aumentando la penalità nel tempo finché rimane in questa posizione.

- **CustomRewardWrapper**: Questo wrapper modifica la ricompensa dell'ambiente penalizzando il robot se il torso si inclina oltre una certa soglia. La penalità aumenta in base al tempo trascorso in questa condizione, scoraggiando comportamenti indesiderati.

#### Creazione dell'Ambiente

Per creare l'ambiente con la ricompensa personalizzata, abbiamo utilizzato una funzione che configura l'ambiente `HalfCheetah-v5` con parametri ottimizzati e applica il `CustomRewardWrapper`.

- **make_env**: Questa funzione crea e restituisce un'istanza dell'ambiente `HalfCheetah-v5` con parametri personalizzati, monitoraggio delle prestazioni e applicazione del `CustomRewardWrapper`.

#### Modifica dell'Architettura della Rete Neurale

Inoltre, abbiamo ottenuto un ulteriore miglioramento modificando l'architettura della rete neurale. Utilizzando il parametro `policy_kwargs`: `dict(net_arch=[256, 256, 128])`, siamo riusciti a ottenere risultati più soddisfacenti per il modello Cheetah. Questa modifica ha permesso di affinare la capacità della rete neurale di apprendere comportamenti più complessi e appropriati per il task.

- **policy_kwargs**: Abbiamo configurato l'architettura della rete neurale con tre strati di dimensioni 256, 256 e 128 neuroni rispettivamente, migliorando la capacità di apprendimento del modello.

In sintesi, attraverso l'introduzione di penalizzazioni mirate e la modifica dell'architettura delle reti neurali, siamo riusciti a migliorare significativamente le prestazioni del modello Cheetah.

### 6.2 Problemi con il modello Ant
Per quanto riguarda il modello Ant, abbiamo esplorato diverse strategie per migliorare le prestazioni. L'ambiente di Ant presenta una maggiore complessità computazionale, richiedendo un'ottimizzazione più accurata dei parametri e dell'architettura della rete neurale. Abbiamo testato vari algoritmi e adattato le strategie di controllo per ottimizzare il comportamento del modello.

In particolare, abbiamo affinato l'architettura della rete neurale e regolato i parametri chiave per gestire la complessità dell'ambiente. Nonostante le difficoltà intrinseche del task, siamo riusciti a individuare e implementare le soluzioni migliori tra quelle testate. Questo ci ha permesso di avanzare significativamente nell'addestramento del modello Ant, ottenendo risultati concreti. Tuttavia, riconosciamo che c'è ancora margine di miglioramento per arrivare a una soluzione ottimale definitiva.

In sintesi, attraverso l'introduzione di penalizzazioni mirate e la modifica dell'architettura delle reti neurali, siamo riusciti a migliorare significativamente le prestazioni dei modelli Cheetah e Ant, sebbene il processo di ottimizzazione continui ad essere un lavoro in corso.