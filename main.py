#!/usr/bin/env python3

import os
import argparse
import torch
import torchaudio
from pyannote.audio import Pipeline
from datetime import timedelta


def format_timestamp(seconds):
    return str(timedelta(seconds=seconds))


def get_audio_energy(audio_path, start, end):
    waveform, sr = torchaudio.load(audio_path)
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    segment = waveform[:, start_sample:end_sample]
    energy = torch.mean(torch.abs(segment)).item()
    return energy


def detect_speech_segments(audio_path, pipeline):
    diarization = pipeline(audio_path)

    if hasattr(diarization, 'speaker_diarization'):
        annotation = diarization.speaker_diarization
    else:
        annotation = diarization

    segments = []
    for turn, _, _ in annotation.itertracks(yield_label=True):
        segments.append({
            'start': turn.start,
            'end': turn.end,
            'duration': turn.end - turn.start
        })

    return segments


def merge_and_attribute(segments_mic1, segments_mic2, audio1_path, audio2_path):
    all_times = set()
    for seg in segments_mic1 + segments_mic2:
        all_times.add(seg['start'])
        all_times.add(seg['end'])

    all_times = sorted(all_times)
    final_segments = []

    for i in range(len(all_times) - 1):
        start = all_times[i]
        end = all_times[i + 1]

        in_mic1 = any(s['start'] <= start < s['end'] for s in segments_mic1)
        in_mic2 = any(s['start'] <= start < s['end'] for s in segments_mic2)

        if in_mic1 or in_mic2:
            energy1 = get_audio_energy(audio1_path, start, end) if in_mic1 else 0
            energy2 = get_audio_energy(audio2_path, start, end) if in_mic2 else 0

            if energy1 > energy2 * 1.2:
                speaker = "Presentateur"
            elif energy2 > energy1 * 1.2:
                speaker = "Invite"
            else:
                speaker = "Overlap"

            if final_segments and final_segments[-1]['speaker'] == speaker:
                final_segments[-1]['end'] = end
                final_segments[-1]['duration'] = end - final_segments[-1]['start']
            else:
                final_segments.append({
                    'start': start,
                    'end': end,
                    'duration': end - start,
                    'speaker': speaker
                })

    return final_segments


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
    parser = argparse.ArgumentParser(description="Diarisation multi-source")
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

    print("Analyse micro presentateur...")
    segments_mic1 = detect_speech_segments(args.presentateur, pipeline)

    print("Analyse micro invite...")
    segments_mic2 = detect_speech_segments(args.invite, pipeline)

    print("Fusion et attribution des locuteurs...")
    final_segments = merge_and_attribute(segments_mic1, segments_mic2,
                                         args.presentateur, args.invite)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_results(final_segments, args.output)

    presentateur_count = sum(1 for s in final_segments if s['speaker'] == 'Presentateur')
    invite_count = sum(1 for s in final_segments if s['speaker'] == 'Invite')
    overlap_count = sum(1 for s in final_segments if s['speaker'] == 'Overlap')

    print(f"\nTermine!")
    print(f"Presentateur: {presentateur_count} segments")
    print(f"Invite: {invite_count} segments")
    print(f"Chevauchements: {overlap_count} segments")


if __name__ == "__main__":
    main()
