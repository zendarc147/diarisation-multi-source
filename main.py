#!/usr/bin/env python3

import argparse
from pathlib import Path
import torch
import torchaudio
import soundfile as sf
import numpy as np
from pyannote.audio import Pipeline
import pympi


# Configuration
NUM_CHANNELS = 2
NUM_SPEAKERS = 2
BUFFER = 0.250
TARGET_SR = 16000


def step1_prepare_audio(source_path, output_dir):
    """
    Step 1: Extract channels and downsample
    """
    print(f"Step 1: Preparation de {source_path}")

    audio, sr = sf.read(source_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared_files = []

    if len(audio.shape) == 2:
        for channel in range(audio.shape[1]):
            channel_audio = audio[:, channel]
            channel_name = f"{'left' if channel == 0 else 'right'}"
            output_path = output_dir / f"{source_path.stem}_{channel_name}.wav"

            audio_tensor = torch.from_numpy(channel_audio).unsqueeze(0).float()
            if sr != TARGET_SR:
                resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
                audio_tensor = resampler(audio_tensor)

            torchaudio.save(str(output_path), audio_tensor, TARGET_SR)
            prepared_files.append((output_path, channel_name))
    else:
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).float()
        if sr != TARGET_SR:
            resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
            audio_tensor = resampler(audio_tensor)

        output_path = output_dir / f"{source_path.stem}_mono.wav"
        torchaudio.save(str(output_path), audio_tensor, TARGET_SR)
        prepared_files.append((output_path, "mono"))

    return prepared_files


def step2_diarize(audio_path, pipeline):
    """
    Step 2: Run diarization pipeline
    """
    print(f"Step 2: Diarisation de {audio_path.name}")

    diarization = pipeline(str(audio_path))

    if hasattr(diarization, 'speaker_diarization'):
        annotation = diarization.speaker_diarization
    else:
        annotation = diarization

    segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append({
            'start': max(0, turn.start - BUFFER),
            'end': turn.end + BUFFER,
            'speaker': speaker
        })

    audio, sr = torchaudio.load(str(audio_path))
    intensities = []
    for seg in segments:
        start_sample = int(seg['start'] * sr)
        end_sample = int(seg['end'] * sr)
        segment_audio = audio[:, start_sample:end_sample]
        intensity = torch.mean(torch.abs(segment_audio)).item()
        intensities.append(intensity)

    mean_intensity = np.mean(intensities) if intensities else 0
    total_duration = sum(s['end'] - s['start'] for s in segments)

    return segments, mean_intensity, total_duration


def step3_combine_outputs(left_data, right_data, output_path, format='eaf'):
    """
    Step 3: Combine diarized outputs into .eaf or .TextGrid
    """
    print(f"Step 3: Combinaison des sorties vers {output_path}")

    left_segs, left_intensity, left_duration = left_data
    right_segs, right_intensity, right_duration = right_data

    if left_duration > right_duration:
        interviewer_segs = (left_segs, "left", left_intensity)
        subject_segs = (right_segs, "right", right_intensity)
    else:
        interviewer_segs = (right_segs, "right", right_intensity)
        subject_segs = (left_segs, "left", left_intensity)

    if format == 'eaf':
        create_eaf(interviewer_segs, subject_segs, output_path)
    else:
        create_textgrid(interviewer_segs, subject_segs, output_path)


def create_eaf(interviewer_data, subject_data, output_path):
    """
    Create ELAN .eaf file
    """
    eaf = pympi.Elan.Eaf()

    int_segs, int_channel, int_intensity = interviewer_data
    subj_segs, subj_channel, subj_intensity = subject_data

    eaf.add_tier('Interviewer_probable')
    eaf.add_tier('Interviewer_unlikely')
    eaf.add_tier('Subject_probable')
    eaf.add_tier('Subject_unlikely')

    for seg in int_segs:
        start_ms = int(seg['start'] * 1000)
        end_ms = int(seg['end'] * 1000)
        eaf.add_annotation('Interviewer_probable', start_ms, end_ms, value='')

    for seg in subj_segs:
        start_ms = int(seg['start'] * 1000)
        end_ms = int(seg['end'] * 1000)
        eaf.add_annotation('Subject_probable', start_ms, end_ms, value='')

    eaf.to_file(str(output_path))


def create_textgrid(interviewer_data, subject_data, output_path):
    """
    Create Praat .TextGrid file
    """
    int_segs, int_channel, int_intensity = interviewer_data
    subj_segs, subj_channel, subj_intensity = subject_data

    all_segs = int_segs + subj_segs
    max_time = max(seg['end'] for seg in all_segs) if all_segs else 10.0

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('File type = "ooTextFile"\n')
        f.write('Object class = "TextGrid"\n\n')
        f.write(f'xmin = 0\n')
        f.write(f'xmax = {max_time}\n')
        f.write('tiers? <exists>\n')
        f.write('size = 2\n')
        f.write('item []:\n')

        for tier_idx, (segs, name) in enumerate([(int_segs, 'Interviewer'), (subj_segs, 'Subject')], 1):
            f.write(f'    item [{tier_idx}]:\n')
            f.write(f'        class = "IntervalTier"\n')
            f.write(f'        name = "{name}"\n')
            f.write(f'        xmin = 0\n')
            f.write(f'        xmax = {max_time}\n')
            f.write(f'        intervals: size = {len(segs)}\n')

            for i, seg in enumerate(segs, 1):
                f.write(f'        intervals [{i}]:\n')
                f.write(f'            xmin = {seg["start"]}\n')
                f.write(f'            xmax = {seg["end"]}\n')
                f.write(f'            text = "*"\n')


def main():
    parser = argparse.ArgumentParser(description="Diarisation multi-source workflow")
    parser.add_argument("--source", required=True, help="Fichier audio source stereo")
    parser.add_argument("--output", default="results", help="Dossier de sortie")
    parser.add_argument("--format", default="eaf", choices=['eaf', 'TextGrid'])
    parser.add_argument("--hf-token", default=None)

    args = parser.parse_args()

    source_path = Path(args.source)
    output_dir = Path(args.output)
    prepared_dir = output_dir / "prepared"

    if not source_path.exists():
        print(f"Fichier source non trouve: {source_path}")
        return

    print("Chargement du modele pyannote...")
    if args.hf_token:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=args.hf_token)
    else:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))

    prepared_files = step1_prepare_audio(source_path, prepared_dir)

    results = {}
    for audio_path, channel in prepared_files:
        segments, intensity, duration = step2_diarize(audio_path, pipeline)
        results[channel] = (segments, intensity, duration)

    if len(results) == 2:
        output_path = output_dir / f"{source_path.stem}.{args.format}"
        step3_combine_outputs(
            results['left'],
            results['right'],
            output_path,
            format=args.format
        )
        print(f"\nTermine! Fichier cree: {output_path}")
    else:
        print("\nAudio mono detecte - pas de combinaison necessaire")
        for channel, (segs, _, _) in results.items():
            print(f"{channel}: {len(segs)} segments")


if __name__ == "__main__":
    main()
