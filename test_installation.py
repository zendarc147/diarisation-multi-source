#!/usr/bin/env python3
"""
Script de test pour vérifier que l'environnement est correctement configuré
"""

import sys

def test_imports():
    """Tester que tous les modules nécessaires sont installés"""
    print("🧪 Test des imports...")

    tests = [
        ("torch", "PyTorch"),
        ("torchaudio", "TorchAudio"),
        ("pyannote.audio", "Pyannote.audio"),
        ("soundfile", "SoundFile"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("librosa", "Librosa"),
        ("matplotlib", "Matplotlib"),
    ]

    failed = []

    for module_name, display_name in tests:
        try:
            __import__(module_name)
            print(f"  ✅ {display_name}")
        except ImportError as e:
            print(f"  ❌ {display_name}: {e}")
            failed.append(display_name)

    return len(failed) == 0


def test_cuda():
    """Vérifier la disponibilité du GPU"""
    print("\n🎮 Test GPU/CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✅ GPU disponible: {torch.cuda.get_device_name(0)}")
            print(f"  📊 Mémoire GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("  ℹ️  Pas de GPU détecté, utilisation du CPU")
        return True
    except Exception as e:
        print(f"  ⚠️  Erreur lors du test GPU: {e}")
        return False


def test_pyannote_auth():
    """Vérifier l'authentification HuggingFace"""
    print("\n🔐 Test authentification HuggingFace...")
    try:
        from huggingface_hub import get_token
        token = get_token()
        if token:
            print("  ✅ Token HuggingFace trouvé")
            print("  💡 N'oubliez pas d'accepter les conditions sur:")
            print("     https://huggingface.co/pyannote/speaker-diarization-3.1")
        else:
            print("  ⚠️  Pas de token HuggingFace")
            print("  💡 Connectez-vous avec: huggingface-cli login")
        return True
    except Exception as e:
        print(f"  ⚠️  Impossible de vérifier l'authentification: {e}")
        print("  💡 Connectez-vous avec: huggingface-cli login")
        return True  # Ne pas bloquer si on ne peut pas vérifier


def test_versions():
    """Afficher les versions des packages principaux"""
    print("\n📦 Versions des packages...")
    try:
        import torch
        import torchaudio
        import pyannote.audio

        print(f"  PyTorch: {torch.__version__}")
        print(f"  TorchAudio: {torchaudio.__version__}")
        print(f"  Pyannote.audio: {pyannote.audio.__version__}")
        return True
    except Exception as e:
        print(f"  ⚠️  Erreur: {e}")
        return False


def main():
    print("=" * 80)
    print("TEST DE L'INSTALLATION - Diarisation Multi-Source")
    print("=" * 80)
    print()

    results = []

    # Test 1: Imports
    results.append(("Imports", test_imports()))

    # Test 2: Versions
    results.append(("Versions", test_versions()))

    # Test 3: GPU/CUDA
    results.append(("GPU", test_cuda()))

    # Test 4: HuggingFace
    results.append(("HuggingFace", test_pyannote_auth()))

    # Résumé
    print("\n" + "=" * 80)
    print("RÉSUMÉ")
    print("=" * 80)

    all_passed = all(result for _, result in results)

    for test_name, passed in results:
        status = "✅" if passed else "❌"
        print(f"{status} {test_name}")

    print()
    if all_passed:
        print("🎉 Tous les tests sont passés!")
        print("🚀 Vous êtes prêt à utiliser le système de diarisation!")
    else:
        print("⚠️  Certains tests ont échoué.")
        print("📖 Consultez le README.md pour les instructions de configuration.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
