# profile_data — `hf-webrtc-voice`

- **scenarios**: `1000` rows in `scenarios.jsonl`
- **kind**: `voice`
- **backend**: `hf`
- **audio**: `audio/q_XXXX.wav` per scenario (question only; generate locally, not in git)
- **format**: single-turn `question` / `answer`

Each JSONL line is one evaluation scenario with expected answer content for profiling / regression checks.
