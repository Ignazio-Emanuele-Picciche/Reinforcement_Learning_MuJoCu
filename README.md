# Reinforcement_Learning_MuJoCu

![Anteprima](videos/halfcheetah.gif)


## Indice 
1. [Descrizione del progetto](#descrizione-del-progetto)
2. [Setup del Enviroment](#setup-del-enviroment)
3. [Pipeline](#pipeline)
4. [Grafici e Risultati](#grafici-e-risultati)

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

## Pipeline

## Grafici e Risultati