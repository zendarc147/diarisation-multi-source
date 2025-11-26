#!/usr/bin/env python3

import argparse
from pathlib import Path
import torch
import torchaudio
from pyannote.audio import Pipeline


BUFFER = 0.250


def get_energy(audio_path, start, end):
    waveform, sr = torchaudio.load(audio_path)
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    segment = waveform[:, start_sample:end_sample]
    return torch.mean(torch.abs(segment)).item()


def detect_segments(audio_path, pipeline):
    diarization = pipeline(str(audio_path))

    if hasattr(diarization, 'speaker_diarization'):
        annotation = diarization.speaker_diarization
    else:
        annotation = diarization

    segments = []
    for turn, _, _ in annotation.itertracks(yield_label=True):
        segments.append({
            'start': max(0, turn.start - BUFFER),
            'end': turn.end + BUFFER
        })

    return segments


def merge_and_attribute(segs_mic1, segs_mic2, audio1, audio2):
    all_times = set()
    for seg in segs_mic1 + segs_mic2:
        all_times.add(seg['start'])
        all_times.add(seg['end'])

    all_times = sorted(all_times)
    final = []

    for i in range(len(all_times) - 1):
        start = all_times[i]
        end = all_times[i + 1]

        in_mic1 = any(s['start'] <= start < s['end'] for s in segs_mic1)
        in_mic2 = any(s['start'] <= start < s['end'] for s in segs_mic2)

        if in_mic1 or in_mic2:
            e1 = get_energy(audio1, start, end) if in_mic1 else 0
            e2 = get_energy(audio2, start, end) if in_mic2 else 0

            if e1 > e2 * 1.2:
                speaker = "Presentateur"
            elif e2 > e1 * 1.2:
                speaker = "Invite"
            else:
                speaker = "Overlap"

            if final and final[-1]['speaker'] == speaker:
                final[-1]['end'] = end
            else:
                final.append({'start': start, 'end': end, 'speaker': speaker})

    return final


def save_results(segments, output):
    with open(output, 'w') as f:
        f.write("DIARISATION\n\n")
        for i, seg in enumerate(segments, 1):
            duration = seg['end'] - seg['start']
            f.write(f"Segment {i:03d} | {seg['speaker']:12} | "
                   f"{seg['start']:7.2f}s - {seg['end']:7.2f}s | {duration:.2f}s\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--presentateur", required=True)
    parser.add_argument("--invite", required=True)
    parser.add_argument("--output", default="results/diarisation.txt")
    parser.add_argument("--hf-token", default=None)

    args = parser.parse_args()

    p1 = Path(args.presentateur)
    p2 = Path(args.invite)

    if not p1.exists() or not p2.exists():
        print("Fichiers non trouves")
        return

    print("Chargement du modele...")
    if args.hf_token:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=args.hf_token)
    else:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))

    print(f"Analyse {p1.name}...")
    segs1 = detect_segments(p1, pipeline)

    print(f"Analyse {p2.name}...")
    segs2 = detect_segments(p2, pipeline)

    print("Fusion et attribution...")
    final = merge_and_attribute(segs1, segs2, p1, p2)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_results(final, args.output)

    stats = {s: sum(1 for seg in final if seg['speaker'] == s)
             for s in ['Presentateur', 'Invite', 'Overlap']}

    print(f"\nTermine! {len(final)} segments")
    print(f"Presentateur: {stats['Presentateur']}, Invite: {stats['Invite']}, "
          f"Overlap: {stats['Overlap']}")


if __name__ == "__main__":
    main()
