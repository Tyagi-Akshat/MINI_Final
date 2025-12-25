from pathlib import Path
import mne

DATA_DIR = Path("data")

for f in sorted(DATA_DIR.glob("S*R*.edf")):
    print("\n==============================")
    print("FILE:", f.name)
    try:
        raw = mne.io.read_raw_edf(f, preload=False, verbose=False)
    except Exception as e:
        print("ERROR reading:", e)
        continue

    print("Channels:", raw.ch_names[:10])
    print("sfreq:", raw.info['sfreq'])

    # Check annotations
    print("Annotations:", raw.annotations)
    try:
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        print("events_from_annotations:", len(events))
        print("event_id:", event_id)
    except Exception as e:
        print("events_from_annotations ERROR:", e)

    # Find stim channel
    stim = [ch for ch in raw.ch_names if "stim" in ch.lower() or "status" in ch.lower()]
    print("Stim-like channels:", stim)

    if stim:
        try:
            ev2 = mne.find_events(raw, stim_channel=stim[0], verbose=False)
            print("find_events:", ev2.shape)
        except Exception as e:
            print("find_events ERROR:", e)
