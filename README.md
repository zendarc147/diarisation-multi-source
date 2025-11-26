# Diarisation Multi-Source

Workflow de diarisation audio pour interviews stereo avec bleeding entre canaux.

## Workflow en 3 étapes

**Step 1**: Extraction des canaux et downsampling
**Step 2**: Diarisation de chaque canal séparément
**Step 3**: Combinaison en fichier .eaf (ELAN) ou .TextGrid (Praat)

## Installation

```bash
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configuration HuggingFace

```bash
huggingface-cli login
```

Accepter les conditions : https://huggingface.co/pyannote/speaker-diarization-3.1

## Utilisation

```bash
# Format ELAN (.eaf)
python main.py --source interview.wav --format eaf

# Format Praat (.TextGrid)
python main.py --source interview.wav --format TextGrid
```

## Structure des sorties

```
results/
├── prepared/
│   ├── interview_left.wav   # Canal gauche 16kHz
│   └── interview_right.wav  # Canal droit 16kHz
└── interview.eaf            # Fichier de sortie
```

## Tiers créés

Pour les fichiers .eaf :
- `Interviewer_probable` : Segments du présentateur
- `Interviewer_unlikely` : Alternative (si inversion détectée)
- `Subject_probable` : Segments de l'invité
- `Subject_unlikely` : Alternative

Le système détermine automatiquement qui parle le plus sur quel canal.

## Paramètres

Modifier dans le code si nécessaire :
- `BUFFER = 0.250` : Extension des segments (secondes)
- `TARGET_SR = 16000` : Fréquence d'échantillonnage cible
