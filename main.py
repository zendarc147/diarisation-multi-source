#!/usr/bin/env python3
"""
Diarisation multi-source pour interviews
Traite deux pistes audio s�par�es (pr�sentateur et invit�)
"""

import os
import argparse
import torch
import torchaudio
from pyannote.audio import Pipeline
from datetime import timedelta


def format_timestamp(seconds):
    """Convertir les secondes en format HH:MM:SS.mmm"""
    return str(timedelta(seconds=seconds))


def process_audio_track(audio_path, pipeline, speaker_label):
    """
    Traiter une piste audio avec pyannote pour d�tecter quand la personne parle

    Args:
        audio_path: Chemin vers le fichier audio
        pipeline: Pipeline pyannote.audio
        speaker_label: Label du locuteur (ex: "Pr�sentateur", "Invit�")

    Returns:
        Liste de segments avec timestamps
    """
    print(f"\n<� Traitement de {speaker_label}: {audio_path}")

    # Charger l'audio
    waveform, sample_rate = torchaudio.load(audio_path)

    # Appliquer la VAD (Voice Activity Detection)
    # Note: Vous aurez besoin d'un token HuggingFace pour utiliser les mod�les
    try:
        diarization = pipeline(audio_path)
    except Exception as e:
        print(f"�  Erreur lors de la diarisation: {e}")
        print("=� Vous devez vous authentifier avec HuggingFace:")
        print("   1. Cr�ez un compte sur https://huggingface.co")
        print("   2. Acceptez les conditions d'utilisation des mod�les pyannote")
        print("   3. G�n�rez un token d'acc�s")
        print("   4. Lancez: huggingface-cli login")
        return []

    # Extraire les segments de parole
    segments = []

    # Pyannote 4.0+ retourne un DiarizeOutput qui contient des Annotations
    try:
        # DiarizeOutput contient speaker_diarization (Annotation)
        if hasattr(diarization, 'speaker_diarization'):
            # Pyannote 4.0+ avec DiarizeOutput
            annotation = diarization.speaker_diarization
        elif hasattr(diarization, 'itertracks'):
            # Ancienne API, c'est déjà un Annotation
            annotation = diarization
        else:
            # Essai de récupération générique
            annotation = diarization

        # Itérer sur les segments
        for turn, _, _ in annotation.itertracks(yield_label=True):
            segments.append({
                'start': turn.start,
                'end': turn.end,
                'duration': turn.end - turn.start,
                'speaker': speaker_label
            })

    except Exception as e:
        print(f"⚠️  Erreur lors de l'extraction des segments: {e}")
        print(f"    Type de l'objet retourné: {type(diarization)}")
        print(f"    Attributs disponibles: {[a for a in dir(diarization) if not a.startswith('_')]}")
        return []

    return segments


def merge_segments(segments_presentateur, segments_invite):
    """
    Fusionner et trier les segments des deux locuteurs
    """
    all_segments = segments_presentateur + segments_invite
    all_segments.sort(key=lambda x: x['start'])
    return all_segments


def save_results(segments, output_path):
    """
    Sauvegarder les r�sultats dans un fichier texte
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("DIARISATION - R�sultats\n")
        f.write("=" * 80 + "\n\n")

        for i, seg in enumerate(segments, 1):
            start_time = format_timestamp(seg['start'])
            end_time = format_timestamp(seg['end'])
            duration = format_timestamp(seg['duration'])

            f.write(f"Segment {i:03d}\n")
            f.write(f"  Locuteur: {seg['speaker']}\n")
            f.write(f"  D�but:    {start_time}\n")
            f.write(f"  Fin:      {end_time}\n")
            f.write(f"  Dur�e:    {duration}\n")
            f.write("-" * 80 + "\n")

    print(f"\n R�sultats sauvegard�s dans: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Diarisation multi-source pour interviews"
    )
    parser.add_argument(
        "--presentateur",
        required=True,
        help="Chemin vers le fichier audio du pr�sentateur"
    )
    parser.add_argument(
        "--invite",
        required=True,
        help="Chemin vers le fichier audio de l'invit�"
    )
    parser.add_argument(
        "--output",
        default="results/diarisation.txt",
        help="Chemin du fichier de sortie"
    )
    parser.add_argument(
        "--hf-token",
        help="Token HuggingFace (optionnel si d�j� connect�)"
    )

    args = parser.parse_args()

    # V�rifier que les fichiers existent
    if not os.path.exists(args.presentateur):
        print(f"L Erreur: Fichier non trouv�: {args.presentateur}")
        return

    if not os.path.exists(args.invite):
        print(f"L Erreur: Fichier non trouv�: {args.invite}")
        return

    print("=� D�marrage de la diarisation multi-source")
    print(f"=� Pr�sentateur: {args.presentateur}")
    print(f"=� Invit�: {args.invite}")

    # Charger le pipeline pyannote
    print("\n=� Chargement du mod�le de diarisation...")
    try:
        if args.hf_token:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=args.hf_token
            )
        else:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1"
            )

        # Utiliser le GPU si disponible
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
            print(" GPU d�tect� et utilis�")
        else:
            print("9 Utilisation du CPU")

    except Exception as e:
        print(f"L Erreur lors du chargement du mod�le: {e}")
        print("\n=� Pour utiliser pyannote.audio, vous devez:")
        print("   1. Accepter les conditions sur: https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("   2. Vous connecter: huggingface-cli login")
        return

    # Traiter les deux pistes
    segments_presentateur = process_audio_track(
        args.presentateur,
        pipeline,
        "Pr�sentateur"
    )

    segments_invite = process_audio_track(
        args.invite,
        pipeline,
        "Invit�"
    )

    # Fusionner les r�sultats
    print("\n= Fusion des segments...")
    all_segments = merge_segments(segments_presentateur, segments_invite)

    print(f"\n=� Statistiques:")
    print(f"   Total segments: {len(all_segments)}")
    print(f"   Segments pr�sentateur: {len(segments_presentateur)}")
    print(f"   Segments invit�: {len(segments_invite)}")

    # Sauvegarder les r�sultats
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_results(all_segments, args.output)

    print("\n( Traitement termin�!")


if __name__ == "__main__":
    main()
