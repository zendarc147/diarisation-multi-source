# Diarisation Multi-Source

Diarisation audio pour interviews avec bleeding entre micros.

## Installation

```bash
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configuration

```bash
huggingface-cli login
```

Accepter les conditions : https://huggingface.co/pyannote/speaker-diarization-3.1

## Utilisation

```bash
python main.py --presentateur mic1.wav --invite mic2.wav
```

Resultat dans `results/diarisation.txt`

## Fonctionnement

1. Detecte les segments de parole sur chaque micro
2. Compare l'energie audio entre les deux micros
3. Attribue chaque segment au bon locuteur
4. Detecte les chevauchements

Le buffer de 0.250s etend chaque segment pour eviter les coupures.
