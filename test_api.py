#!/usr/bin/env python3
"""
Script pour tester l'API pyannote.audio 4.0 sans fichiers audio
"""

print("🔍 Exploration de l'API pyannote.audio 4.0")
print("=" * 80)

# Importer les modules
try:
    from pyannote.audio import Pipeline
    from pyannote.core import Annotation, Segment
    print("✅ Imports réussis")
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    exit(1)

# Examiner l'objet Annotation
print("\n📋 Méthodes de l'objet Annotation:")
annotation_methods = [m for m in dir(Annotation) if not m.startswith('_') and callable(getattr(Annotation, m, None))]
for method in sorted(annotation_methods)[:20]:  # Afficher les 20 premières
    print(f"  - {method}")

# Créer un exemple d'annotation pour tester
print("\n🧪 Test avec une annotation d'exemple:")
annotation = Annotation()
annotation[Segment(0, 5), 0] = "speaker1"
annotation[Segment(5, 10), 1] = "speaker2"

print("\n✓ Annotation créée avec 2 segments")
print(f"  Nombre de segments: {len(list(annotation.itertracks()))}")

# Tester l'itération
print("\n🔄 Test d'itération sur les segments:")
for i, (segment, track, label) in enumerate(annotation.itertracks(yield_label=True)):
    print(f"  Segment {i+1}:")
    print(f"    Début: {segment.start}s")
    print(f"    Fin: {segment.end}s")
    print(f"    Label: {label}")

print("\n✅ L'API fonctionne correctement!")
print("\n💡 Pour votre code, utilisez:")
print("   for segment, track, label in diarization.itertracks(yield_label=True):")
print("       # Traiter segment.start, segment.end, etc.")
