Requires torch and optionally numpy and matplotlib

```powershell
python -m VoiceSpaceClassifier.train PATH_TO_AUDIO --template=speaker_template
python -m VoiceSpaceClassifier.get_similarity_matrix PATH_TO_MODEL PATH_TO_AUDIO --template=speaker_template
```

See help for other args