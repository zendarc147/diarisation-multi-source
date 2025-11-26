#!/usr/bin/env python3

import os
import argparse
import torch
from pyannote.audio import Pipeline
from datetime import timedelta


def format_timestamp(seconds):
    return str(timedelta(seconds=seconds))


def process_audio_track(audio_path, pipeline, speaker_label):
    print(f"Traitement de {speaker_label}...")

    diarization = pipeline(audio_path)
    segments = []

    if hasattr(diarization, 'speaker_diarization'):
        annotation = diarization.speaker_diarization
    else:
        annotation = diarization

    for turn, _, _ in annotation.itertracks(yield_label=True):
        segments.append({
            'start': turn.start,
            'end': turn.end,
            'duration': turn.end - turn.start,
            'speaker': speaker_label
        })

    return segments


def save_results(segments, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("DIARISATION - Resultats\n")
        f.write("=" * 80 + "\n\n")

        for i, seg in enumerate(segments, 1):
            f.write(f"Segment {i:03d}\n")
            f.write(f"  Locuteur: {seg['speaker']}\n")
            f.write(f"  Debut:    {format_timestamp(seg['start'])}\n")
            f.write(f"  Fin:      {format_timestamp(seg['end'])}\n")
            f.write(f"  Duree:    {format_timestamp(seg['duration'])}\n")
            f.write("-" * 80 + "\n")

    print(f"Resultats sauvegardes: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Diarisation audio")
    parser.add_argument("--presentateur", required=True)
    parser.add_argument("--invite", required=True)
    parser.add_argument("--output", default="results/diarisation.txt")
    parser.add_argument("--hf-token", default=None)

    args = parser.parse_args()

    if not os.path.exists(args.presentateur) or not os.path.exists(args.invite):
        print("Fichiers audio non trouves")
        return

    print("Chargement du modele...")
    if args.hf_token:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=args.hf_token)
    else:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))

    segments_presentateur = process_audio_track(args.presentateur, pipeline, "Presentateur")
    segments_invite = process_audio_track(args.invite, pipeline, "Invite")

    all_segments = segments_presentateur + segments_invite
    all_segments.sort(key=lambda x: x['start'])

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_results(all_segments, args.output)

    print(f"Termine! {len(all_segments)} segments trouves")


if __name__ == "__main__":
    main()
