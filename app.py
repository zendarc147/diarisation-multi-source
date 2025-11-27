#!/usr/bin/env python3

import gradio as gr
import torch
from pyannote.audio import Pipeline
from pathlib import Path
import tempfile


BUFFER = 0.5
pipeline = None


def calc_energie(audio_path, debut, fin):
    import torchaudio
    son, sr = torchaudio.load(audio_path)
    debut_sample = int(debut * sr)
    fin_sample = int(fin * sr)
    seg = son[:, debut_sample:fin_sample]
    return torch.mean(torch.abs(seg)).item()


def energie_globale(audio_path):
    import torchaudio
    son, _ = torchaudio.load(audio_path)
    return torch.mean(torch.abs(son)).item()


def detecter_segments(audio_path, pipeline):
    diarization = pipeline(str(audio_path))

    if hasattr(diarization, 'speaker_diarization'):
        annotation = diarization.speaker_diarization
    else:
        annotation = diarization

    segs = []
    for turn, _, _ in annotation.itertracks(yield_label=True):
        segs.append({'start': max(0, turn.start - BUFFER), 'end': turn.end + BUFFER})

    return segs


def fusionner(segs_mic1, segs_mic2, audio1, audio2):
    energie_moy1 = energie_globale(audio1)
    energie_moy2 = energie_globale(audio2)

    tous_temps = set()
    for seg in segs_mic1 + segs_mic2:
        tous_temps.add(seg['start'])
        tous_temps.add(seg['end'])

    tous_temps = sorted(tous_temps)
    resultat = []

    for i in range(len(tous_temps) - 1):
        debut = tous_temps[i]
        fin = tous_temps[i + 1]

        dans_mic1 = any(s['start'] <= debut < s['end'] for s in segs_mic1)
        dans_mic2 = any(s['start'] <= debut < s['end'] for s in segs_mic2)

        if dans_mic1 or dans_mic2:
            e1 = calc_energie(audio1, debut, fin) if dans_mic1 else 0
            e2 = calc_energie(audio2, debut, fin) if dans_mic2 else 0

            e1_norm = e1 / energie_moy1 if energie_moy1 > 0 else 0
            e2_norm = e2 / energie_moy2 if energie_moy2 > 0 else 0

            if e1_norm > e2_norm * 1.5:
                qui = "Presentateur"
            elif e2_norm > e1_norm * 1.5:
                qui = "Invite"
            else:
                qui = "Overlap"

            if resultat and resultat[-1]['speaker'] == qui:
                resultat[-1]['end'] = fin
            else:
                resultat.append({'start': debut, 'end': fin, 'speaker': qui})

    return resultat


def traiter(audio_pres, audio_inv):
    global pipeline

    if audio_pres is None or audio_inv is None:
        return "Veuillez uploader les deux fichiers audio"

    if pipeline is None:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))

    segs1 = detecter_segments(audio_pres, pipeline)
    segs2 = detecter_segments(audio_inv, pipeline)
    final = fusionner(segs1, segs2, audio_pres, audio_inv)

    stats = {s: sum(1 for seg in final if seg['speaker'] == s)
             for s in ['Presentateur', 'Invite', 'Overlap']}

    resultat_texte = "DIARISATION\n\n"
    for i, seg in enumerate(final, 1):
        duree = seg['end'] - seg['start']
        resultat_texte += f"Segment {i:03d} | {seg['speaker']:12} | "
        resultat_texte += f"{seg['start']:7.2f}s - {seg['end']:7.2f}s | {duree:.2f}s\n"

    resultat_texte += f"\n{len(final)} segments - "
    resultat_texte += f"P:{stats['Presentateur']} I:{stats['Invite']} O:{stats['Overlap']}"

    return resultat_texte


interface = gr.Interface(
    fn=traiter,
    inputs=[
        gr.Audio(type="filepath", label="Audio Presentateur"),
        gr.Audio(type="filepath", label="Audio Invite")
    ],
    outputs=gr.Textbox(label="Resultats", lines=20),
    title="Diarisation Multi-Source",
    description="Upload les deux fichiers audio (presentateur et invite)"
)


if __name__ == "__main__":
    interface.launch()
