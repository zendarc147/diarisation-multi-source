#!/bin/bash
# Script d'activation de l'environnement virtuel

echo "🚀 Activation de l'environnement virtuel..."
source venv/bin/activate

echo "✅ Environnement activé!"
echo ""
echo "📋 Commandes utiles:"
echo "  python main.py --help              # Voir l'aide"
echo "  pip list                           # Voir les packages installés"
echo "  huggingface-cli login              # Se connecter à HuggingFace"
echo "  deactivate                         # Désactiver l'environnement"
echo ""
