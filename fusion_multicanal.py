#!/usr/bin/env python3

import torch
import torchaudio
from pyannote.audio import Pipeline
from pathlib import Path


def fusionner_multicanal(fichier_invite, fichier_presentateur, fichier_sortie):
    """Fusionne deux fichiers mono en un fichier stéréo multicanal"""

    # Charger les deux fichiers audio
    audio_invite, sr_invite = torchaudio.load(fichier_invite)
    audio_pres, sr_pres = torchaudio.load(fichier_presentateur)

    # Vérifier que les sample rates sont identiques
    assert sr_invite == sr_pres, "Les sample rates doivent être identiques"

    # Convertir en mono si nécessaire
    if audio_invite.shape[0] > 1:
        audio_invite = torch.mean(audio_invite, dim=0, keepdim=True)
    if audio_pres.shape[0] > 1:
        audio_pres = torch.mean(audio_pres, dim=0, keepdim=True)

    # Adapter la longueur (prendre le minimum)
    longueur_min = min(audio_invite.shape[1], audio_pres.shape[1])
    audio_invite = audio_invite[:, :longueur_min]
    audio_pres = audio_pres[:, :longueur_min]

    # Fusionner en stéréo : canal 0 = invité, canal 1 = présentateur
    audio_multicanal = torch.cat([audio_invite, audio_pres], dim=0)

    # Sauvegarder
    torchaudio.save(fichier_sortie, audio_multicanal, sr_invite)
    print(f"Fichier multicanal créé: {fichier_sortie}")
    print(f"  - Canal 0: Invité")
    print(f"  - Canal 1: Présentateur")
    print(f"  - Shape: {audio_multicanal.shape}")

    return fichier_sortie, sr_invite


def diariser_multicanal(fichier_multicanal):
    """Effectue la diarisation sur le fichier multicanal"""

    # Charger le pipeline pyannote
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))

    # Lancer la diarisation
    print(f"\nDiarisation en cours sur {fichier_multicanal}...")
    diarization = pipeline(fichier_multicanal)

    # Extraire les résultats
    if hasattr(diarization, 'speaker_diarization'):
        annotation = diarization.speaker_diarization
    else:
        annotation = diarization

    # Afficher les résultats
    print("\n=== RÉSULTATS DIARISATION MULTICANAL ===\n")

    segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append({
            'start': turn.start,
            'end': turn.end,
            'speaker': speaker,
            'duration': turn.end - turn.start
        })

    for i, seg in enumerate(segments, 1):
        print(f"Segment {i:03d} | {seg['speaker']:10} | "
              f"{seg['start']:7.2f}s - {seg['end']:7.2f}s | "
              f"{seg['duration']:.2f}s")

    # Statistiques
    speakers = set(seg['speaker'] for seg in segments)
    print(f"\n{len(segments)} segments | {len(speakers)} locuteurs détectés")

    return segments


if __name__ == "__main__":
    # Chemins des fichiers
    fichier_invite = "audio_input/invite.wav"
    fichier_presentateur = "audio_input/presentateur.wav"
    fichier_multicanal = "multicanal_fusion.wav"

    # Étape 1: Fusionner en multicanal
    fusionner_multicanal(fichier_invite, fichier_presentateur, fichier_multicanal)

    # Étape 2: Diariser
    diariser_multicanal(fichier_multicanal)
