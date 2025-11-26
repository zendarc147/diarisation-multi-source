# Configuration du Token HuggingFace

Pour utiliser les modèles pyannote.audio, vous devez obtenir un token HuggingFace.

## Étape 1 : Créer un compte HuggingFace

1. Allez sur https://huggingface.co
2. Cliquez sur "Sign Up" pour créer un compte gratuit
3. Validez votre email

## Étape 2 : Accepter les conditions d'utilisation

Vous devez accepter les conditions pour ces modèles :

1. **Speaker Diarization** : https://huggingface.co/pyannote/speaker-diarization-3.1
   - Cliquez sur "Agree and access repository"

2. **Segmentation Model** : https://huggingface.co/pyannote/segmentation-3.0
   - Cliquez sur "Agree and access repository"

## Étape 3 : Obtenir votre token

1. Allez sur https://huggingface.co/settings/tokens
2. Cliquez sur "New token"
3. Donnez un nom à votre token (ex: "diarisation-project")
4. Choisissez le type : **"Read"** (suffisant pour télécharger les modèles)
5. Cliquez sur "Generate token"
6. **Copiez le token** (il ressemble à : `hf_xxxxxxxxxxxxxxxxxxxxx`)

⚠️ **Important** : Conservez ce token en lieu sûr ! Vous ne pourrez plus le voir après avoir fermé la page.

## Étape 4 : Configurer le token

### Option A : Via la CLI (Recommandé)

```bash
source venv/bin/activate
huggingface-cli login
```

Collez votre token quand demandé.

### Option B : Variable d'environnement

1. Copiez le fichier exemple :
   ```bash
   cp .env.example .env
   ```

2. Éditez `.env` et remplacez `votre_token_huggingface_ici` par votre vrai token

3. Le token sera chargé automatiquement

### Option C : En ligne de commande

```bash
python main.py \
    --presentateur audio_input/presentateur.wav \
    --invite audio_input/invite.wav \
    --hf-token "hf_votre_token_ici"
```

## Vérification

Pour vérifier que tout fonctionne :

```bash
source venv/bin/activate
python test_installation.py
```

Vous devriez voir : `✅ Token HuggingFace trouvé`

## Dépannage

### "You must accept the user conditions"

Assurez-vous d'avoir accepté les conditions sur :
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

### "Invalid token"

1. Vérifiez que le token est bien copié (pas d'espaces avant/après)
2. Vérifiez que le token commence par `hf_`
3. Générez un nouveau token si nécessaire

### "Token not found"

Si vous utilisez la CLI, reconnectez-vous :
```bash
huggingface-cli logout
huggingface-cli login
```
