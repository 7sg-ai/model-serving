# profile_data — `nim-webrtc-voice`

- **scenarios**: `1000` rows in `scenarios.jsonl`
- **kind**: `voice`
- **backend**: `nim`
- **audio**: `audio/q_XXXX.wav` per scenario (question only; generate locally, not in git)
- **format**: single-turn `question` / `answer`

Each JSONL line is one evaluation scenario with expected answer content for profiling / regression checks.
