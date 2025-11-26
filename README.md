# Diarisation Multi-Source

Projet de diarisation audio pour interviews avec deux pistes séparées.

## Installation

```bash
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Créer un compte HuggingFace et accepter les conditions :
- https://huggingface.co/pyannote/speaker-diarization-3.1

Se connecter :
```bash
huggingface-cli login
```

## Utilisation

```bash
python main.py --presentateur audio1.wav --invite audio2.wav
```

Le résultat sera dans `results/diarisation.txt`

## Options

- `--presentateur` : fichier audio du présentateur
- `--invite` : fichier audio de l'invité
- `--output` : fichier de sortie (défaut: results/diarisation.txt)
- `--hf-token` : token HuggingFace (optionnel)
