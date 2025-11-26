# Diarisation Multi-Source pour Interviews

Projet de diarisation audio pour interviews avec deux pistes séparées (présentateur et invité).

## Structure du Projet

```
diaz_V2/
├── venv/                 # Environnement virtuel Python
├── audio_input/          # Placez vos fichiers audio ici
├── data/
│   ├── presentateur/     # Données spécifiques au présentateur
│   └── invite/          # Données spécifiques à l'invité
├── results/             # Résultats de la diarisation
├── main.py              # Script principal
├── requirements.txt     # Dépendances Python
└── README.md           # Ce fichier
```

## Installation

L'environnement est déjà configuré ! Les dépendances installées :

- **PyTorch 2.8.0** (avec support CUDA)
- **Pyannote.audio 4.0.2** (diarisation de locuteurs)
- **TorchAudio** (traitement audio)
- **Librosa** (analyse audio)
- **Matplotlib** (visualisation)

## Configuration HuggingFace

Pour utiliser les modèles pyannote, vous devez :

1. Créer un compte sur [HuggingFace](https://huggingface.co)
2. Accepter les conditions d'utilisation du modèle :
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
3. Se connecter :
   ```bash
   source venv/bin/activate
   huggingface-cli login
   ```

## Utilisation

### 1. Activer l'environnement virtuel

```bash
source venv/bin/activate
```

### 2. Préparer vos fichiers audio

Placez vos deux fichiers audio (un pour le présentateur, un pour l'invité) dans le dossier `audio_input/`.

Formats supportés : WAV, MP3, FLAC, OGG, etc.

### 3. Lancer la diarisation

```bash
python main.py \
    --presentateur audio_input/presentateur.wav \
    --invite audio_input/invite.wav \
    --output results/diarisation.txt
```

### Options disponibles

- `--presentateur` : Chemin vers le fichier audio du présentateur (obligatoire)
- `--invite` : Chemin vers le fichier audio de l'invité (obligatoire)
- `--output` : Chemin du fichier de sortie (défaut: `results/diarisation.txt`)
- `--hf-token` : Token HuggingFace si vous n'êtes pas connecté via CLI

### Exemple complet

```bash
# Activer l'environnement
source venv/bin/activate

# Lancer le traitement
python main.py \
    --presentateur audio_input/host.wav \
    --invite audio_input/guest.wav \
    --output results/interview_2024.txt
```

## Format de Sortie

Le fichier de résultats contient :

```
DIARISATION - Résultats
================================================================================

Segment 001
  Locuteur: Présentateur
  Début:    0:00:00.123000
  Fin:      0:00:05.456000
  Durée:    0:00:05.333000
--------------------------------------------------------------------------------
Segment 002
  Locuteur: Invité
  Début:    0:00:05.789000
  Fin:      0:00:12.345000
  Durée:    0:00:06.556000
--------------------------------------------------------------------------------
...
```

## Langues Étrangères

Le système fonctionne avec n'importe quelle langue ! La diarisation (détection de qui parle quand) est basée sur les caractéristiques vocales, pas sur le contenu linguistique.

## Fonctionnement

1. **Voice Activity Detection (VAD)** : Détecte quand une personne parle sur chaque piste
2. **Segmentation** : Identifie les segments de parole
3. **Fusion** : Combine les résultats des deux pistes avec leurs labels respectifs
4. **Export** : Génère un fichier texte avec tous les segments ordonnés chronologiquement

## Ressources Supplémentaires

### Documentation
- [Pyannote.audio Documentation](https://github.com/pyannote/pyannote-audio)
- [HuggingFace Hub](https://huggingface.co/pyannote)
- [PyTorch Audio](https://pytorch.org/audio/)

### Modèles Utilisés
- `pyannote/speaker-diarization-3.1` : Pipeline complet de diarisation
- Basé sur des modèles d'embedding vocaux et de segmentation

## Dépannage

### Erreur : "Import could not be resolved"
C'est normal dans l'IDE. Activez l'environnement virtuel :
```bash
source venv/bin/activate
```

### Erreur : "You must accept the user conditions"
Visitez https://huggingface.co/pyannote/speaker-diarization-3.1 et acceptez les conditions.

### GPU non détecté
Si vous avez un GPU NVIDIA mais que le système utilise le CPU, vérifiez :
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Développé avec

- Python 3.13.7
- PyTorch 2.8.0 (CUDA 12.8)
- Pyannote.audio 4.0.2
